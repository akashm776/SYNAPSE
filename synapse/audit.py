"""Bounded A/M-only diagnostics. The historical llm engine is never modified."""
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
from torch import nn

from .llm.core import Generator, finite_gradients
from .llm.data import digest, scheduled_batch
from .llm.experiment import (atomic_json, atomic_torch, cpu, digest_tensor_tree,
    file_hash, hardware, load_checkpoint, read_json, restore, run_lock, snapshot,
    source_hash, validate_config)
from .llm.model import (check_prompt_causality, load_learner, load_tokenizer,
    make_optimizer, meta_objective, objective)


DEFAULTS = dict(learner_seed=789, states=[100, 500], pairs=2, repeats=2,
    epsilons=[0.0001, 0.001], fit_steps=20, fit_lr=0.001, selection_seed=61005,
    generator_seed=91005, max_seconds=1800)


def settings(supplied):
    if set(supplied)-set(DEFAULTS):
        raise ValueError('Unknown audit settings')
    c = {**deepcopy(DEFAULTS), **supplied}
    for key in ('pairs', 'repeats', 'fit_steps'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'Invalid {key}')
    if c['repeats'] < 2:
        raise ValueError('Repeatability requires at least two repeats')
    for key in ('learner_seed', 'selection_seed', 'generator_seed'):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f'Invalid {key}')
    if not c['states'] or c['states'] != sorted(set(c['states'])) or any(type(s) is not int or s < 1 for s in c['states']):
        raise ValueError('Invalid states')
    if not c['epsilons'] or len(set(c['epsilons'])) != len(c['epsilons']) or any(not math.isfinite(e) or not 0 < e < .25 for e in c['epsilons']):
        raise ValueError('Perturbations must be distinct and between zero and .25')
    if any(not math.isfinite(c[k]) or c[k] <= 0 for k in ('fit_lr', 'max_seconds')):
        raise ValueError('Invalid fitting rate or wall budget')
    return c


class Probe(nn.Module):
    arm = 'learned_barycenter'

    def __init__(self, generator=None, epsilon=0.):
        super().__init__()
        self.generator, self.epsilon = generator, epsilon
        self.features = self.weights = None

    def forward(self, features):
        if self.generator is None:
            w = features.new_tensor([.25+self.epsilon, .25-self.epsilon, .25, .25]).expand(features.shape[:2])
        else:
            w = self.generator(features)
        self.features, self.weights = features.detach().cpu(), w.detach().cpu()
        return w

    def diagnostics(self):
        return {'weights': self.weights.tolist(),
            'max_deviation_from_uniform': float((self.weights-.25).abs().max()),
            'candidate_feature_std': self.features.std(dim=1, unbiased=False).mean(0).tolist(),
            'feature_min': self.features.flatten(0, 1).min(0).values.tolist(),
            'feature_max': self.features.flatten(0, 1).max(0).values.tolist()}


def vector(tensors, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten()
                      for g, p in zip(tensors, parameters)])


def difference(a, b):
    # b is the named reference. Relative error alone is misleading near zero.
    delta, na, nb = a-b, float(a.norm()), float(b.norm())
    return {'l2': float(delta.norm()), 'max_abs': float(delta.abs().max()),
            'reference_l2': nb, 'relative_l2': float(delta.norm())/max(nb, 1e-30),
            'cosine': float(torch.dot(a, b)/(na*nb)) if na and nb else None,
            'bitwise_equal': bool(torch.equal(a, b))}


def set_precision(learner, precision):
    """Use identical bf16-rounded frozen weights; only arithmetic precision varies."""
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]
    with torch.no_grad():
        for p in learner.parameters():
            if not p.requires_grad:
                p.data = p.data.to(dtype)
        for b in learner.buffers():
            if b.is_floating_point():
                b.data = b.data.to(dtype)
    assert all(p.dtype == torch.float32 for p in learner.trainable().values())


def trial(learner, optimizer, parent, inner, outer, tokenizer, config, probe):
    restore(learner, optimizer, parent)
    params = tuple(learner.trainable().values())
    native, _ = objective(learner, inner, tokenizer, config)
    ng = torch.autograd.grad(native, params, allow_unused=True)
    native_gradient = vector(ng, params)
    del native, ng
    if probe is None:
        total, metrics = objective(learner, inner, tokenizer, config)
    else:
        total, metrics = objective(learner, inner, tokenizer, config, probe.arm, probe)
    gradients = torch.autograd.grad(total, params, allow_unused=True)
    if not finite_gradients(gradients):
        raise FloatingPointError('Nonfinite audit gradient')
    full_gradient = vector(gradients, params)
    for p, g in zip(params, gradients):
        p.grad = g
    norm = torch.nn.utils.clip_grad_norm_(params, config['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    delta = torch.cat([(p.detach().cpu()-parent['parameters'][name]).flatten()
                      for name,p in learner.trainable().items()])
    with torch.no_grad():
        loss, _ = objective(learner, outer, tokenizer, config)
    aux_gradient = full_gradient-native_gradient
    result = {'inner': metrics, 'outer_M_loss': float(loss), 'gradient_norm': float(norm),
        'weighted_aux_gradient_norm': float(aux_gradient.norm()), 'update_norm': float(delta.norm()),
        'aux_to_native_gradient_ratio': float(aux_gradient.norm())/max(float(native_gradient.norm()), 1e-30)}
    if probe is not None:
        result.update(probe.diagnostics())
    restore(learner, optimizer, parent)
    return result, {'aux_gradient':aux_gradient, 'update':delta}


def comparisons(current, reference):
    return {k:difference(current[k], reference[k]) for k in current}


def write_report(output, results):
    rows = [r for g in results['repeatability'] for r in g['records'] if 'condition' in r]
    repeated = [r for r in rows if 'vs_first_repeat' in r]
    identical = sum(all(v['bitwise_equal'] for v in r['vs_first_repeat'].values())
                    and r['M_difference_vs_first_repeat']==0 for r in repeated)
    lines = ['# SYNAPSE numerical and generator-fitting audit', '',
        'A/M-only diagnostic. No D/test evaluation, no generalization claim.', '',
        f'Bitwise-identical repeated gradient/update/M-loss comparisons: {identical}/{len(repeated)}.', '',
        'The precision comparison promotes the same bf16-rounded frozen weights to fp32; it does not recover full-precision pretrained weights.', '',
        '| Teacher state | Arithmetic | Fixed-set virtual M loss before | After | Change |',
        '|---|---|---:|---:|---:|']
    for fit in results['fits']:
        lines.append(f"| {fit['state']} | {fit['precision']} | {fit['initial']['mean_M_virtual_loss']:.9g} | {fit['final']['mean_M_virtual_loss']:.9g} | {fit['fixed_set_loss_change']:+.9g} |")
    lines += ['', 'See audit_summary.json for auxiliary-gradient and actual-update comparisons against uniform weights, repeated trials, and bf16, plus per-example weights and feature variation.', '',
        'A negative fixed-set change only demonstrates fitting on the same A/M examples. Different episodes are not independent evaluation replications. Do not select a new efficacy recipe from these diagnostics alone.']
    path = output/'AUDIT_REPORT.md'
    temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    os.replace(temporary,path)


def repeatability_group(learner, optimizer, parent, pairs, pair_index, tokenizer, config, audit, trained):
    inner, outer = pairs[pair_index]
    parent_digest = digest_tensor_tree(parent)
    records, refs = [], {}
    for precision in ('bf16', 'fp32'):
        set_precision(learner, precision)
        restore(learner, optimizer, parent)
        causality = check_prompt_causality(learner, inner, tokenizer)
        conditions = [('native',None), ('uniform',Probe()), ('learned',Probe(trained))]
        conditions += [(f'epsilon_{e:g}',Probe(epsilon=e)) for e in audit['epsilons']]
        for name, probe in conditions:
            for repeat in range(audit['repeats']):
                row, vectors = trial(learner, optimizer, parent, inner, outer, tokenizer, config, probe)
                row.update(precision=precision, condition=name, repeat=repeat)
                row['M_delta_vs_native'] = row['outer_M_loss']-refs.get((precision,'native'), (row,None))[0]['outer_M_loss']
                if repeat:
                    ref, refvec = refs[precision,name]
                    row['vs_first_repeat'] = comparisons(vectors, refvec)
                    row['M_difference_vs_first_repeat'] = row['outer_M_loss']-ref['outer_M_loss']
                else:
                    refs[precision,name] = (deepcopy(row),vectors)
                if name not in ('native','uniform'):
                    ref, refvec = refs[precision,'uniform']
                    row['vs_uniform'] = comparisons(vectors, refvec)
                    row['M_difference_vs_uniform'] = row['outer_M_loss']-ref['outer_M_loss']
                if precision == 'fp32':
                    ref, refvec = refs['bf16',name]
                    row['vs_bf16_first_repeat'] = comparisons(vectors, refvec)
                    row['M_difference_vs_bf16'] = row['outer_M_loss']-ref['outer_M_loss']
                records.append(row)
        records.append({'precision':precision, 'same_shape_causality':causality})
    assert digest_tensor_tree(parent) == parent_digest
    after = snapshot(learner,optimizer)
    assert all(digest_tensor_tree(after[k]) == digest_tensor_tree(parent[k]) for k in ('parameters','optimizer'))
    return {'pair_index':pair_index,'records':records}


def assess_fit(learner, optimizer, parent, pairs, tokenizer, config, generator):
    values = []
    for inner, outer in pairs:
        restore(learner,optimizer,parent)
        probe = Probe(generator)
        loss, _ = meta_objective(learner,optimizer,inner,outer,tokenizer,config,probe.arm,probe)
        values.append({'M_virtual_loss':float(loss.detach()), **probe.diagnostics()})
        del loss
    return {'mean_M_virtual_loss':sum(v['M_virtual_loss'] for v in values)/len(values), 'pairs':values}


class Pause(Exception):
    pass


def safe_paths(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Audit output must be separate from, not inside/above, the source run')
    return source, output


def load_source(source, audit):
    # This is read-only. Never call the pilot prepare/execute functions here.
    manifest = read_json(source/'manifest.json')
    config = validate_config(read_json(source/'protocol.json'))
    if manifest['status'] not in ('development_complete','complete') or not read_json(source/'preflight.json')['passed']:
        raise ValueError('Use a completed, preflight-passed source training run')
    if config != manifest['identity']['config'] or manifest['identity']['source_sha256'] != source_hash():
        raise ValueError('Source protocol/engine mismatch')
    triples = [s for s in config['seed_triples'] if s[2]==audit['learner_seed']]
    if len(triples)!=1 or not set(audit['states']).issubset(config['states']):
        raise ValueError('Unknown pipeline seed or teacher states')
    data = read_json(source/'data.json')
    if digest(data) != manifest['data_sha256']:
        raise ValueError('Source prepared data hash mismatch')
    # B/D/test are never passed to a model, sampled, or exported.
    parts = {k:data['partitions'][k] for k in ('A','M')}
    del data
    if min(map(len,parts.values())) < audit['pairs']:
        raise ValueError('Too few distinct A/M examples for fixed pairs')
    pairs = [(scheduled_batch(parts['A'],i,1,audit['selection_seed']),
              scheduled_batch(parts['M'],i,1,audit['selection_seed']+1)) for i in range(audit['pairs'])]
    paths = ['manifest.json','protocol.json','preflight.json']
    prefix = f"seed_{audit['learner_seed']}"
    paths += [f'{prefix}/teacher/step_{s}.pt' for s in audit['states']]
    paths += [f'{prefix}/generators/learned_barycenter.pt']
    hashes = {p:file_hash(source/p) for p in paths}
    return config, manifest, pairs, hashes, triples[0]


def run_audit(source, output, supplied, max_units=None):
    source, output = safe_paths(source,output)
    audit = settings(supplied)
    if max_units is not None and max_units < 1:
        raise ValueError('max-units must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    config, manifest, pairs, hashes, triple = load_source(source,audit)
    device_name = hardware(config)
    identity = {'settings':audit, 'source_files_sha256':hashes, 'data_sha256':manifest['data_sha256'],
        'pilot_source_sha256':source_hash(), 'audit_source_sha256':file_hash(Path(__file__)),
        'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','numpy')},
        'python':platform.python_version(), 'device':device_name,
        'precision_design':'bf16-rounded frozen base; bf16 versus promoted-fp32 arithmetic; fp32 adapters and moments',
        'tf32':False,'deterministic_algorithms':True,
        'pairs':[{'A_id':i[0]['id'],'M_id':o[0]['id']} for i,o in pairs]}
    units = 0
    def boundary():
        if time.perf_counter()-started >= audit['max_seconds'] or (max_units is not None and units >= max_units):
            raise Pause('Budget reached at safe boundary. Rerun the same audit to resume.')
    with run_lock(output):
        path = output/'audit_manifest.json'
        if path.exists() and read_json(path)['identity'] != identity:
            raise ValueError('Audit inputs/settings/environment changed; use a new audit directory')
        atomic_json(path,{'identity':identity,'status':'running','source_run':str(source),
                          'test_evaluated':False,'development_evaluated':False,'generalization_evidence':False})
        learner = optimizer = None
        try:
            boundary()
            tokenizer = load_tokenizer(config,manifest['resolved'])
            learner = load_learner(config,manifest['resolved'],triple[0])
            # In tiny tests too, both conditions start from the same bf16 rounding.
            set_precision(learner,'bf16')
            optimizer = make_optimizer(learner,config)
            saved = load_checkpoint(source/f"seed_{audit['learner_seed']}/generators/learned_barycenter.pt")
            trained = Generator('learned_barycenter').to(config['device'])
            trained.load_state_dict(saved['generator'])
            trained.requires_grad_(False)
            for state in audit['states']:
                parent = load_checkpoint(source/f"seed_{audit['learner_seed']}/teacher/step_{state}.pt")['state']
                for pair_index in range(audit['pairs']):
                    target = output/f'state_{state}/repeatability_pair_{pair_index}.json'
                    if target.exists():
                        continue
                    boundary()
                    before = time.perf_counter()
                    if config['device']=='cuda':
                        torch.cuda.reset_peak_memory_stats()
                    result = repeatability_group(learner,optimizer,parent,pairs,pair_index,tokenizer,config,audit,trained)
                    result.update(state=state,seconds=time.perf_counter()-before,
                        peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30 if config['device']=='cuda' else None)
                    atomic_json(target,result)
                    units += 1
                    print(f'Audit state={state} pair={pair_index}: both precisions saved',flush=True)
                for precision in ('bf16','fp32'):
                    set_precision(learner,precision)
                    target = output/f'state_{state}/fit_{precision}.pt'
                    torch.manual_seed(audit['generator_seed'])
                    generator = Generator('learned_barycenter').to(config['device'])
                    fitting = torch.optim.AdamW(generator.parameters(),lr=audit['fit_lr'],weight_decay=0.,foreach=False)
                    start, history = 0, []
                    if target.exists():
                        saved = load_checkpoint(target)
                        generator.load_state_dict(saved['generator'])
                        fitting.load_state_dict(saved['optimizer'])
                        start,history,initial = saved['step'],saved['history'],saved['initial']
                    else:
                        boundary()
                        initial = assess_fit(learner,optimizer,parent,pairs,tokenizer,config,generator)
                    for step in range(start,audit['fit_steps']):
                        boundary()
                        inner,outer = pairs[step % len(pairs)]
                        restore(learner,optimizer,parent)
                        fitting.zero_grad(set_to_none=True)
                        probe = Probe(generator)
                        loss, _ = meta_objective(learner,optimizer,inner,outer,tokenizer,config,probe.arm,probe)
                        grads = torch.autograd.grad(loss,tuple(generator.parameters()),allow_unused=True)
                        if not finite_gradients(grads):
                            raise FloatingPointError('Nonfinite fitting gradient')
                        teacher_after = snapshot(learner,optimizer)
                        if any(digest_tensor_tree(teacher_after[k])!=digest_tensor_tree(parent[k]) for k in ('parameters','optimizer')):
                            raise RuntimeError('Meta-fitting mutated teacher parameters or moments')
                        del teacher_after
                        before = vector(tuple(generator.parameters()),tuple(generator.parameters()))
                        for p,g in zip(generator.parameters(),grads):
                            p.grad=g
                        norm = torch.nn.utils.clip_grad_norm_(generator.parameters(),1.,error_if_nonfinite=True)
                        fitting.step()
                        after = vector(tuple(generator.parameters()),tuple(generator.parameters()))
                        history.append({'step':step+1,'pair_index':step%len(pairs),'M_virtual_loss_before':float(loss.detach()),
                            'generator_gradient_norm':float(norm),'parameter_update_norm':float((after-before).norm()),
                            **probe.diagnostics()})
                        del loss, grads
                        atomic_torch(target,{'step':step+1,'generator':cpu(generator.state_dict()),
                            'optimizer':cpu(fitting.state_dict()),'history':history,'initial':initial})
                        units += 1
                    report = target.with_suffix('.json')
                    if not report.exists():
                        boundary()
                        final = assess_fit(learner,optimizer,parent,pairs,tokenizer,config,generator)
                        atomic_json(report,{'state':state,'precision':precision,'initial':initial,'final':final,
                            'fixed_set_loss_change':final['mean_M_virtual_loss']-initial['mean_M_virtual_loss'],
                            'history':history,'is_generalization_result':False})
                        units += 1
                        print(f'Audit state={state} precision={precision}: fixed-set fit saved',flush=True)
                    del generator,fitting
            # A second input check detects source writes during this diagnostic.
            if any(file_hash(source/name)!=h for name,h in hashes.items()):
                raise ValueError('Source files changed during audit')
            results = {'repeatability':[read_json(p) for p in sorted(output.glob('state_*/repeatability_*.json'))],
                       'fits':[read_json(p) for p in sorted(output.glob('state_*/fit_*.json'))],
                       'test_evaluated':False,'generalization_evidence':False}
            atomic_json(output/'audit_summary.json',results)
            write_report(output,results)
            status='complete'
        except Pause as exc:
            status='paused'
            print(str(exc),flush=True)
        except Exception as exc:
            current=read_json(path)
            atomic_json(path,{**current,'status':'failed','error':f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            del learner,optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        current=read_json(path)
        atomic_json(path,{**current,'status':status,'last_invocation_seconds':time.perf_counter()-started})
        return read_json(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--max-units',type=int)
    args=parser.parse_args()
    print(run_audit(args.source,args.output,read_json(args.config),args.max_units)['status'])


if __name__=='__main__':
    main()
