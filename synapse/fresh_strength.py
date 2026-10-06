"""Strength replication on diagnostic-fresh A/M pairs; no held-out efficacy claim."""
import argparse
from copy import deepcopy
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import time

import torch

from . import audit, learnability, strength
from .llm.experiment import atomic_json, file_hash, hardware, load_checkpoint, read_json, restore, run_lock, source_hash
from .llm.model import load_learner, load_tokenizer, make_optimizer


DEFAULTS = dict(learner_seed=789,states=[100,500],pairs=16,alphas=[0.,.01,.1,1.],max_seconds=1800)
SELECTION_SEED = 61005
EXCLUDED_PREFIX = 2


def settings(supplied):
    if set(supplied)-set(DEFAULTS):
        raise ValueError('Unknown fresh-strength settings')
    c={**deepcopy(DEFAULTS),**supplied}
    strength.settings({**c,'selection_seed':SELECTION_SEED})
    if c['pairs']>32:
        raise ValueError('This bounded replication supports at most 32 fresh pairs')
    return c


def load_source(source,c):
    # The preceding diagnostics used the first two no-replacement samples with
    # this seed. Continue that same permutation, excluding both component IDs.
    config,manifest,all_pairs,hashes,triple=audit.load_source(source,
        {**c,'selection_seed':SELECTION_SEED,'pairs':c['pairs']+EXCLUDED_PREFIX})
    excluded,pairs=all_pairs[:EXCLUDED_PREFIX],all_pairs[EXCLUDED_PREFIX:]
    forbidden={r[0]['id'] for pair in excluded for r in pair}
    ids=[r[0]['id'] for pair in pairs for r in pair]
    if len(ids)!=len(set(ids)) or forbidden.intersection(ids):
        raise ValueError('Fresh pair IDs are duplicated or overlap the excluded diagnostic examples')
    return config,manifest,pairs,hashes,triple,excluded


def recipes():
    return strength.recipes()[:5]  # Uniform plus all four vertices; no winners selected.


def group(learner,optimizer,parent,pair,tokenizer,config,alpha):
    config={**config,'alpha':alpha}
    restore(learner,optimizer,parent)
    causality=audit.check_prompt_causality(learner,pair[0],tokenizer)
    buffers=audit.buffer_fingerprint(learner)
    rows,refs=[],{}
    for name,weights in [('native',None),('native_repeat',None),*recipes(),('uniform_repeat',[.25]*4)]:
        probe=strength.StrengthProbe(learnability.FixedWeights(weights)) if weights is not None else None
        row,vectors=audit.trial(learner,optimizer,parent,*pair,tokenizer,config,probe)
        learnability.assert_parent(learner,optimizer,parent)
        if not all(math.isfinite(v) for v in [row['outer_M_loss'],row['gradient_norm'],row['update_norm'],
                *[r['virtual_M_loss'] for r in row['virtual_real_parity'].values()]]):
            raise FloatingPointError('Nonfinite fresh-strength result')
        row.update(recipe=name,specified_weights=weights,auxiliary_active=weights is not None and alpha>0,
            clipped=row['gradient_norm']>config['max_grad_norm'],
            clip_factor=min(1.,config['max_grad_norm']/(row['gradient_norm']+1e-6)))
        if name in ('native','uniform'):
            refs[name]=(row,vectors)
        native,nv=refs['native']
        row['vs_native']=audit.comparisons(vectors,nv)
        row['M_real_delta_vs_native']=row['outer_M_loss']-native['outer_M_loss']
        row['M_virtual_delta_vs_native']=row['virtual_real_parity']['meta_path']['virtual_M_loss']-native['virtual_real_parity']['meta_path']['virtual_M_loss']
        if weights is not None:
            uniform,uv=refs['uniform']
            row['vs_uniform']=audit.comparisons(vectors,uv)
            row['M_real_delta_vs_uniform']=row['outer_M_loss']-uniform['outer_M_loss']
        if name.endswith('_repeat'):
            ref,rv=refs[name.removesuffix('_repeat')]
            row['vs_first_repeat']=audit.comparisons(vectors,rv)
            row['M_real_repeat_difference']=row['outer_M_loss']-ref['outer_M_loss']
        if alpha==0 and (row['M_real_delta_vs_native']!=0 or row['M_virtual_delta_vs_native']!=0 or
                         not all(v['bitwise_equal'] for v in row['vs_native'].values())):
            raise RuntimeError('Zero strength differs from native')
        rows.append(row)
    if audit.buffer_fingerprint(learner)!=buffers:
        raise RuntimeError('Model buffers changed')
    return dict(alpha=alpha,records=rows,same_shape_causality=causality,buffers_unchanged=True)


def numerical_reference(rows):
    return max(max(abs(r['virtual_real_parity'][kind]['M_virtual_minus_real'])
                   for r in rows for kind in ('same_gradient','meta_path')),
               max(abs(r['M_real_repeat_difference']) for r in rows if 'M_real_repeat_difference' in r),
               8*torch.finfo(torch.float32).eps*max(1.,max(abs(r['outer_M_loss']) for r in rows)))


def describe(real,virtual,refs):
    n=len(real)
    if not n or len(virtual)!=n or len(refs)!=n:
        raise ValueError('Mismatched contrast arrays')
    leave=[mean(real[:i]+real[i+1:]) for i in range(n)] if n>1 else []
    return dict(mean_real=mean(real),median_real=median(real),mean_virtual=mean(virtual),
        helpful_pairs=sum(v<0 for v in real),harmful_pairs=sum(v>0 for v in real),zero_pairs=sum(v==0 for v in real),
        reductions_exceeding_reference=sum(v < -r for v,r in zip(real,refs)),
        increases_exceeding_reference=sum(v > r for v,r in zip(real,refs)),
        leave_one_pair_out_mean_range=[min(leave),max(leave)] if leave else None,
        per_pair_real=real,per_pair_virtual=virtual,per_pair_reference=refs)


def summarize(groups,c):
    summaries=[]
    for state in c['states']:
        for alpha in c['alphas']:
            selected=sorted([g for g in groups if g['state']==state and g['alpha']==alpha],key=lambda g:g['pair_index'])
            if [g['pair_index'] for g in selected]!=list(range(c['pairs'])):
                raise ValueError('Missing or duplicate fresh pair groups')
            indexed=[{r['recipe']:r for r in g['records']} for g in selected]
            refs=[numerical_reference(g['records']) for g in selected]
            contrasts={}
            for name,_ in recipes():
                real=[r[name]['M_real_delta_vs_native'] for r in indexed]
                virtual=[r[name]['M_virtual_delta_vs_native'] for r in indexed]
                contrasts[name+'_vs_native']=describe(real,virtual,refs)
            # Average losses after four SEPARATE one-candidate updates; this is
            # not a mixture, an ensemble prediction, or a best-candidate oracle.
            real=[r['uniform']['outer_M_loss']-mean(r[f'vertex_{i}']['outer_M_loss'] for i in range(4)) for r in indexed]
            virtual=[r['uniform']['virtual_real_parity']['meta_path']['virtual_M_loss']-
                     mean(r[f'vertex_{i}']['virtual_real_parity']['meta_path']['virtual_M_loss'] for i in range(4)) for r in indexed]
            contrasts['uniform_vs_average_single']=describe(real,virtual,refs)
            summaries.append(dict(state=state,alpha=alpha,contrasts=contrasts,
                uniform_mean_aux_native_gradient_ratio=mean(r['uniform']['aux_to_native_gradient_ratio'] for r in indexed),
                uniform_clipped_pairs=sum(r['uniform']['clipped'] for r in indexed)))
    return dict(settings=c,groups=groups,summaries=summaries,test_evaluated=False,
        development_evaluated=False,generalization_evidence=False,selected_alpha=None,selected_recipe=None)


def write_report(output,result):
    lines=['# SYNAPSE diagnostic-fresh strength replication','','Fresh relative to the two previously inspected A/M pairs, not historically untouched training examples.',
        '', 'One seed; states share the same pairs. This is a one-step sensitivity replication, not a fine-tuning efficacy test. Negative deltas favor the named treatment.', '']
    for contrast,title in [('uniform_vs_native','Uniform versus native'),('uniform_vs_average_single','Uniform versus average single-candidate update')]:
        lines += ['## '+title,'','| State | Alpha | Mean real delta | Median | Mean virtual delta | Helpful/harmful pairs | Beyond reference: help/harm | Leave-one-pair-out mean range |',
                  '|---|---:|---:|---:|---:|---|---|---|']
        for s in result['summaries']:
            r=s['contrasts'][contrast]
            leave=r['leave_one_pair_out_mean_range']
            span=f'[{leave[0]:+.7g}, {leave[1]:+.7g}]' if leave else 'n/a'
            lines.append(f"| {s['state']} | {s['alpha']:g} | {r['mean_real']:+.7g} | {r['median_real']:+.7g} | {r['mean_virtual']:+.7g} | {r['helpful_pairs']}/{r['harmful_pairs']} | {r['reductions_exceeding_reference']}/{r['increases_exceeding_reference']} | {span} |")
        lines.append('')
    lines += ['The average-single control averages losses after four separate fixed single-candidate updates. It is not a selected winner or a combined update. Individual vertex contrasts are in fresh_strength_summary.json.', '',
        'Per-pair numerical reference = maximum measured virtual/real loss gap, repeat drift, or eight fp32 epsilons times loss scale. Beyond-reference counts are descriptive only: not error bounds, confidence intervals, or significance tests. Raw sign counts include tiny numerical changes.', '',
        'Inspect medians, sign counts, clipping, virtual/real agreement and leave-one-pair-out means alongside the mean. Pairs are not pipeline-seed replications. The default three positive strengths were chosen after inspecting the previous diagnostic; no alpha or recipe is selected here. No generator is trained, and no B/D/test data is evaluated.']
    path=output/'FRESH_STRENGTH_REPORT.md'
    temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    os.replace(temporary,path)


def run(source,output,supplied,max_units=None):
    source,output=audit.safe_paths(source,output)
    c=settings(supplied)
    if max_units is not None and (type(max_units) is not int or max_units<1):
        raise ValueError('max-units must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    config,manifest,pairs,hashes,triple,excluded=load_source(source,c)
    if config['alpha'] not in c['alphas']:
        raise ValueError('Include the original source strength')
    packages=['torch','transformers','numpy']
    if config['backend']=='hf':
        packages+=['datasets','huggingface-hub','tokenizers','accelerate']
    pair_ids=lambda data:[dict(A_id=i[0]['id'],M_id=o[0]['id']) for i,o in data]
    identity=dict(version=1,settings=c,selection_seed=SELECTION_SEED,excluded_pairs=pair_ids(excluded),pairs=pair_ids(pairs),
        source_files_sha256=hashes,data_sha256=manifest['data_sha256'],pilot_source_sha256=source_hash(),
        diagnostic_files_sha256={p.name:file_hash(p) for p in (Path(__file__),Path(audit.__file__),Path(learnability.__file__),Path(strength.__file__))},
        versions={k:importlib.metadata.version(k) for k in packages},python=platform.python_version(),
        device=hardware(config),tf32=False,deterministic_algorithms=True,source_alpha=config['alpha'],
        precision='bf16-rounded frozen parameters promoted to fp32; fp32 adapters/moments; original buffers',
        recipes=[dict(name=n,weights=w) for n,w in recipes()])
    units=0
    def boundary():
        if time.perf_counter()-started>=c['max_seconds'] or (max_units is not None and units>=max_units):
            raise audit.Pause('Budget reached at saved group boundary; rerun same code/config/output to resume.')
    with run_lock(output):
        path=output/'fresh_strength_manifest.json'
        if path.exists() and read_json(path)['identity']!=identity:
            raise ValueError('Fresh-strength inputs/settings/environment changed; use a new directory')
        if path.exists() and read_json(path)['status']=='complete':
            return read_json(path)
        atomic_json(path,dict(identity=identity,status='running',source_run=str(source),test_evaluated=False,
            development_evaluated=False,generalization_evidence=False))
        learner=optimizer=None
        try:
            boundary()
            tokenizer=load_tokenizer(config,manifest['resolved'])
            learner=load_learner(config,manifest['resolved'],triple[0])
            buffers=audit.buffer_fingerprint(learner)
            buffer_path=output/'model_buffers.json'
            if buffer_path.exists() and read_json(buffer_path)!=buffers:
                raise ValueError('Loaded buffers differ from prior invocation')
            atomic_json(buffer_path,buffers)
            audit.set_precision(learner,'bf16')
            audit.set_precision(learner,'fp32')
            optimizer=make_optimizer(learner,config)
            for state in c['states']:
                parent=load_checkpoint(source/f"seed_{c['learner_seed']}/teacher/step_{state}.pt")['state']
                for i,pair in enumerate(pairs):
                    for j,alpha in enumerate(c['alphas']):
                        target=output/f'state_{state}/pair_{i}_alpha_{j}.json'
                        if target.exists():
                            continue
                        boundary()
                        if audit.buffer_fingerprint(learner)!=buffers:
                            raise RuntimeError('Model buffers changed')
                        before=time.perf_counter()
                        result=group(learner,optimizer,parent,pair,tokenizer,config,alpha)
                        result.update(state=state,pair_index=i,seconds=time.perf_counter()-before)
                        atomic_json(target,result)
                        units+=1
                        print(f'State {state}, fresh pair {i+1}/{c["pairs"]}, alpha {alpha:g}: saved',flush=True)
            if any(file_hash(source/name)!=h for name,h in hashes.items()):
                raise ValueError('Source files changed during diagnostic')
            if audit.buffer_fingerprint(learner)!=buffers:
                raise RuntimeError('Model buffers changed')
            groups=[read_json(output/f'state_{state}/pair_{i}_alpha_{j}.json')
                    for state in c['states'] for i in range(c['pairs']) for j in range(len(c['alphas']))]
            result=summarize(groups,c)
            atomic_json(output/'fresh_strength_summary.json',result)
            write_report(output,result)
            status='complete'
        except audit.Pause as exc:
            status='paused'
            print(str(exc),flush=True)
        except Exception as exc:
            atomic_json(path,{**read_json(path),'status':'failed','error':f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            del learner,optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_json(path,{**read_json(path),'status':status,'last_invocation_seconds':time.perf_counter()-started})
        return read_json(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','config'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-units',type=int)
    args=parser.parse_args()
    print(run(args.source,args.output,read_json(args.config),args.max_units)['status'])


if __name__=='__main__':
    main()
