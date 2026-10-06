"""A/M-only fixed-mixture auxiliary-strength sensitivity, not recipe selection."""
import argparse
from copy import deepcopy
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import time

import torch

from . import audit, learnability
from .llm.experiment import (atomic_json, file_hash, hardware, load_checkpoint,
    read_json, restore, run_lock, source_hash)
from .llm.model import load_learner, load_tokenizer, make_optimizer


DEFAULTS = dict(learner_seed=789, states=[100,500], pairs=2,
    selection_seed=61005, alphas=[0.,.001,.01,.1,1.], max_seconds=1800)


def settings(supplied):
    if set(supplied)-set(DEFAULTS):
        raise ValueError('Unknown strength settings')
    c = {**deepcopy(DEFAULTS), **supplied}
    audit.settings({k:c[k] for k in audit.DEFAULTS if k in c})
    a = c['alphas']
    if (not isinstance(a,list) or len(a)<2 or
        any(type(v) not in (int,float) or not math.isfinite(v) or v<0 or v>1 for v in a) or
        a != sorted(set(a)) or a[0] != 0):
        raise ValueError('Use distinct sorted alphas in [0,1], including zero and a positive value')
    return c


def recipes():
    return [('uniform',[.25]*4)] + [
        (f'vertex_{i}',[float(k==i) for k in range(4)]) for i in range(4)] + [
        (f'edge_{i}_{j}',[.5 if k in (i,j) else 0. for k in range(4)])
        for i in range(4) for j in range(i+1,4)]


class StrengthProbe(audit.Probe):
    def diagnostics(self):
        # alpha=0 exits objective before invoking the generator. No measured
        # weights/features exist in this case; never fabricate them.
        return {} if self.weights is None else super().diagnostics()


def group(learner, optimizer, parent, pair, tokenizer, config, alpha):
    # Never mutate the source protocol. Native evaluation of M has no auxiliary.
    treatment_config = {**config,'alpha':alpha}
    restore(learner,optimizer,parent)
    causality = audit.check_prompt_causality(learner,pair[0],tokenizer)
    buffers = audit.buffer_fingerprint(learner)
    records, refs = [], {}
    conditions = [('native',None),('native_repeat',None),*recipes(),('uniform_repeat',[.25]*4)]
    for name, weights in conditions:
        # Exercise the actual learned-arm alpha=0 early exit, not just native.
        probe = StrengthProbe(learnability.FixedWeights(weights)) if weights is not None else None
        row, vectors = audit.trial(learner,optimizer,parent,*pair,tokenizer,treatment_config,probe)
        learnability.assert_parent(learner,optimizer,parent)
        if not all(math.isfinite(x) for x in [row['outer_M_loss'],row['gradient_norm'],row['update_norm'],
                *[v['virtual_M_loss'] for v in row['virtual_real_parity'].values()]]):
            raise FloatingPointError('Nonfinite strength diagnostic')
        row.update(recipe=name, specified_weights=weights, auxiliary_active=probe is not None and alpha>0,
            clipped=row['gradient_norm']>config['max_grad_norm'],
            clip_factor=min(1.,config['max_grad_norm']/(row['gradient_norm']+1e-6)))
        if name in ('native','uniform'):
            refs[name] = (row,vectors)
        native,native_vectors = refs['native']
        row['vs_native'] = audit.comparisons(vectors,native_vectors)
        row['M_real_delta_vs_native'] = row['outer_M_loss']-native['outer_M_loss']
        row['M_virtual_delta_vs_native'] = row['virtual_real_parity']['meta_path']['virtual_M_loss']-native['virtual_real_parity']['meta_path']['virtual_M_loss']
        if weights is not None:
            uniform,uniform_vectors = refs['uniform']
            row['vs_uniform'] = audit.comparisons(vectors,uniform_vectors)
            row['M_real_delta_vs_uniform'] = row['outer_M_loss']-uniform['outer_M_loss']
            row['M_virtual_delta_vs_uniform'] = row['virtual_real_parity']['meta_path']['virtual_M_loss']-uniform['virtual_real_parity']['meta_path']['virtual_M_loss']
        if name.endswith('_repeat'):
            ref,ref_vectors = refs[name.removesuffix('_repeat')]
            row['vs_first_repeat'] = audit.comparisons(vectors,ref_vectors)
            row['M_real_repeat_difference'] = row['outer_M_loss']-ref['outer_M_loss']
        if alpha == 0 and (row['M_real_delta_vs_native'] != 0 or row['M_virtual_delta_vs_native'] != 0 or
                           not all(v['bitwise_equal'] for v in row['vs_native'].values())):
            raise RuntimeError('Zero-strength control is not exactly native')
        records.append(row)
    if audit.buffer_fingerprint(learner) != buffers:
        raise RuntimeError('Strength diagnostic changed model buffers')
    return dict(alpha=alpha, records=records, same_shape_causality=causality, buffers_unchanged=True)


def summarize(groups, c, source_alpha):
    summaries = []
    for state in c['states']:
        for alpha in c['alphas']:
            selected = [g for g in groups if g['state']==state and g['alpha']==alpha]
            assert len(selected) == c['pairs']
            rows = [r for g in selected for r in g['records']]
            parity = max(abs(r['virtual_real_parity'][k]['M_virtual_minus_real'])
                         for r in rows for k in ('same_gradient','meta_path'))
            repeat = max(abs(r['M_real_repeat_difference']) for r in rows if 'M_real_repeat_difference' in r)
            floor = 8*torch.finfo(torch.float32).eps*max(1.,max(abs(r['outer_M_loss']) for r in rows))
            by_recipe = []
            for name,_ in recipes():
                treatments = [r for r in rows if r['recipe']==name]
                mean = lambda key: sum(r[key] for r in treatments)/len(treatments)
                by_recipe.append(dict(recipe=name,
                    mean_real_delta_vs_native=mean('M_real_delta_vs_native'),
                    mean_virtual_delta_vs_native=mean('M_virtual_delta_vs_native'),
                    mean_real_delta_vs_uniform=mean('M_real_delta_vs_uniform'),
                    mean_virtual_delta_vs_uniform=mean('M_virtual_delta_vs_uniform'),
                    mean_aux_native_gradient_ratio=mean('aux_to_native_gradient_ratio'),
                    clipped_pairs=sum(r['clipped'] for r in treatments),
                    per_pair_real_delta_vs_native=[r['M_real_delta_vs_native'] for r in treatments]))
            summaries.append(dict(state=state,alpha=alpha,recipes=by_recipe,
                numerical_reference=dict(max_abs_virtual_real_gap=parity,max_abs_repeat_drift=repeat,
                    eight_epsilon_loss_scale=floor,reference_scale=max(parity,repeat,floor),is_error_bound=False)))
    return dict(settings=c,source_alpha=source_alpha,groups=groups,summaries=summaries,
        test_evaluated=False,development_evaluated=False,generalization_evidence=False,
        selected_alpha=None,selected_recipe=None)


def write_report(output, result):
    lines = ['# SYNAPSE fixed-mixture strength sensitivity', '',
        'A/M-only, one restored learner update per condition. No generator fitting, held-out evaluation, or recipe selection.', '',
        f"Original auxiliary strength: {result['source_alpha']}. Frozen weights retain bf16 rounding; arithmetic is fp32, TF32 off, model buffers unchanged.", '',
        'Negative loss deltas are lower M loss. The recipe range averages each fixed recipe across pairs first; it is not an average of per-pair winners.', '',
        '| State | Alpha | Uniform real Δ vs native | Uniform virtual Δ vs native | Fixed-recipe mean real Δ range vs uniform | Uniform aux/native gradient ratio | Uniform clipped pairs | Numerical reference |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for s in result['summaries']:
        u=s['recipes'][0]
        values=[r['mean_real_delta_vs_uniform'] for r in s['recipes']]
        lines.append(f"| {s['state']} | {s['alpha']:g} | {u['mean_real_delta_vs_native']:+.9g} | {u['mean_virtual_delta_vs_native']:+.9g} | [{min(values):+.9g}, {max(values):+.9g}] | {u['mean_aux_native_gradient_ratio']:.6g} | {u['clipped_pairs']}/{result['settings']['pairs']} | {s['numerical_reference']['reference_scale']:.9g} |")
    lines += ['', 'The numerical reference is the largest observed virtual/real loss gap, repeat drift, or eight fp32 epsilons times loss scale. It is a descriptive heuristic, not a statistical threshold or error bound. It does not bound a treatment-minus-control difference.', '',
        'Alpha zero must match native exactly for all recipes. The native update is recomputed and repeated in each group; uniform is repeated too. JSON includes all per-pair losses, gradient/update contrasts, clipping, repeat checks, and virtual/real parity. Compare actual and simulated changes, not only the magnitude of the auxiliary gradient.', '',
        'A larger auxiliary gradient can change clipping and Adam update direction without improving M loss. Fixed-mixture sensitivity need not imply a learnable or transferable generator. This scan cannot establish that the original intervention was too weak, that a stronger one helps fine-tuning, or that any recipe generalizes. Only one seed/two fixed A/M pairs are used by default. No winning alpha or recipe is selected.']
    path=output/'STRENGTH_REPORT.md'
    temp=path.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines)+'\n')
    os.replace(temp,path)


def run(source, output, supplied, max_units=None):
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
    config,manifest,pairs,hashes,triple=audit.load_source(source,c)
    if config['alpha'] not in c['alphas']:
        raise ValueError('The alpha grid must include the original source strength')
    packages=['torch','transformers','numpy']
    if config['backend']=='hf':
        packages+=['datasets','huggingface-hub','tokenizers','accelerate']
    identity=dict(version=1,settings=c,source_files_sha256=hashes,data_sha256=manifest['data_sha256'],
        pilot_source_sha256=source_hash(),diagnostic_files_sha256={
            p.name:file_hash(p) for p in (Path(__file__),Path(audit.__file__),Path(learnability.__file__))},
        versions={k:importlib.metadata.version(k) for k in packages},python=platform.python_version(),
        device=hardware(config),tf32=False,deterministic_algorithms=True,source_alpha=config['alpha'],
        precision='bf16-rounded frozen parameters promoted to fp32; fp32 adapters/moments; original buffers',
        recipes=[dict(name=n,weights=w) for n,w in recipes()],
        pairs=[dict(A_id=i[0]['id'],M_id=o[0]['id']) for i,o in pairs])
    units=0
    def boundary():
        if time.perf_counter()-started>=c['max_seconds'] or (max_units is not None and units>=max_units):
            raise audit.Pause('Budget reached at a saved group boundary; rerun the same code/config/output to resume.')
    with run_lock(output):
        path=output/'strength_manifest.json'
        if path.exists() and read_json(path)['identity']!=identity:
            raise ValueError('Strength inputs/settings/environment changed; use a new output directory')
        if path.exists() and read_json(path)['status']=='complete':
            return read_json(path)
        atomic_json(path,dict(identity=identity,status='running',source_run=str(source),
            test_evaluated=False,development_evaluated=False,generalization_evidence=False))
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
                for index,pair in enumerate(pairs):
                    for alpha_index,alpha in enumerate(c['alphas']):
                        target=output/f'state_{state}/pair_{index}_alpha_{alpha_index}.json'
                        if target.exists():
                            continue
                        boundary()
                        if audit.buffer_fingerprint(learner)!=buffers:
                            raise RuntimeError('Model buffers changed')
                        before=time.perf_counter()
                        result=group(learner,optimizer,parent,pair,tokenizer,config,alpha)
                        result.update(state=state,pair_index=index,seconds=time.perf_counter()-before)
                        atomic_json(target,result)
                        units+=1
                        print(f'State {state}, pair {index}, alpha {alpha:g}: saved',flush=True)
            if any(file_hash(source/name)!=h for name,h in hashes.items()):
                raise ValueError('Source files changed during diagnostic')
            if audit.buffer_fingerprint(learner)!=buffers:
                raise RuntimeError('Model buffers changed')
            groups=[read_json(output/f'state_{state}/pair_{i}_alpha_{j}.json')
                    for state in c['states'] for i in range(c['pairs']) for j in range(len(c['alphas']))]
            result=summarize(groups,c,config['alpha'])
            atomic_json(output/'strength_summary.json',result)
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
