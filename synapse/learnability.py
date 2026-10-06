"""Fixed-A/M fp32 learnability diagnostic; never evaluates B, D, or test."""
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

from . import audit
from .llm.core import Generator, finite_gradients
from .llm.experiment import (atomic_json, atomic_torch, cpu, digest_tensor_tree,
    file_hash, hardware, load_checkpoint, read_json, restore, run_lock, snapshot,
    source_hash)
from .llm.model import load_learner, load_tokenizer, make_optimizer, meta_objective


DEFAULTS = dict(learner_seed=789, states=[100, 500], pairs=2, fit_steps=50,
    fit_lr=.01, adam_eps=1e-8, selection_seed=61005, generator_seed=91005,
    max_seconds=1800)
ARMS = ('shared_logits', 'pair_logits', 'feature_scorer')


def settings(supplied):
    if set(supplied)-set(DEFAULTS):
        raise ValueError('Unknown learnability settings')
    c = {**deepcopy(DEFAULTS), **supplied}
    # Share the audit's seed/state/pair validation without changing its defaults.
    audit.settings({k:c[k] for k in audit.DEFAULTS if k in c})
    if not math.isfinite(c['adam_eps']) or c['adam_eps'] <= 0:
        raise ValueError('Invalid Adam epsilon')
    return c


class DirectWeights(nn.Module):
    """Raw candidate slots, NOT hardness ranks; pair weights cannot transfer."""
    arm = 'learned_barycenter'

    def __init__(self, count):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(count, 4))
        self.pair_index = 0

    def forward(self, features):
        if features.shape[0] != 1 or features.shape[1] != 4:
            raise ValueError('Fixed-pair diagnostic requires one row and four candidates')
        index = 0 if len(self.logits) == 1 else self.pair_index
        return self.logits[index:index+1].softmax(-1)


class FixedWeights(nn.Module):
    arm = 'learned_barycenter'

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, features):
        return self.weights.to(features).expand(features.shape[:2])


def make_generator(arm, config, device):
    torch.manual_seed(config['generator_seed'])
    if arm not in ARMS:
        raise ValueError(arm)
    model = Generator('learned_barycenter') if arm == 'feature_scorer' else DirectWeights(
        config['pairs'] if arm == 'pair_logits' else 1)
    return model.to(device)


def use_pair(generator, index):
    if isinstance(generator, DirectWeights):
        generator.pair_index = index


def assert_parent(learner, optimizer, parent):
    after = snapshot(learner, optimizer)
    if any(digest_tensor_tree(after[k]) != digest_tensor_tree(parent[k])
           for k in ('parameters', 'optimizer')):
        raise RuntimeError('Diagnostic mutated teacher parameters or moments')


def measure(learner, optimizer, parent, pair, tokenizer, config, generator):
    row, _ = audit.trial(learner, optimizer, parent, *pair, tokenizer, config,
                         audit.Probe(generator))
    losses = [row['outer_M_loss']] + [p['virtual_M_loss'] for p in row['virtual_real_parity'].values()]
    if not all(math.isfinite(v) for v in losses):
        raise FloatingPointError('Nonfinite diagnostic M loss')
    assert_parent(learner, optimizer, parent)
    return row


def landscape(learner, optimizer, parent, pair, tokenizer, config):
    recipes = [('uniform', [.25]*4), ('uniform_repeat', [.25]*4)]
    recipes += [(f'vertex_{i}', [float(j == i) for j in range(4)]) for i in range(4)]
    recipes += [(f'edge_{i}_{j}', [.5 if k in (i,j) else 0. for k in range(4)])
                for i in range(4) for j in range(i+1,4)]
    records = []
    for name, weights in recipes:
        row = measure(learner, optimizer, parent, pair, tokenizer, config, FixedWeights(weights))
        records.append({'recipe':name, **row})
    return records


def assess(learner, optimizer, parent, pairs, tokenizer, config, generator):
    records = []
    for index, pair in enumerate(pairs):
        use_pair(generator, index)
        records.append(measure(learner, optimizer, parent, pair, tokenizer, config, generator))
    return {'pairs':records,
        'mean_M_real_loss':sum(r['outer_M_loss'] for r in records)/len(records),
        'mean_M_virtual_loss':sum(r['virtual_real_parity']['meta_path']['virtual_M_loss']
                                  for r in records)/len(records)}


def fit_step(learner, optimizer, parent, pairs, tokenizer, config, generator, fitting):
    """Equal full-fixed-set steps for all arms; one graph resident at a time."""
    fitting.zero_grad(set_to_none=True)
    params = tuple(generator.parameters())
    before = audit.vector(params, params)
    records = []
    for index, (inner, outer) in enumerate(pairs):
        restore(learner, optimizer, parent)
        use_pair(generator, index)
        probe = audit.Probe(generator)
        loss, _ = meta_objective(learner, optimizer, inner, outer, tokenizer, config, probe.arm, probe)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fitting M loss')
        grads = torch.autograd.grad(loss/len(pairs), params, allow_unused=True)
        if not finite_gradients(grads):
            raise FloatingPointError('Nonfinite generator gradient')
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g.detach().clone() if p.grad is None else p.grad + g.detach()
        records.append({'M_virtual_loss_before':float(loss.detach()), **probe.diagnostics()})
        del loss, grads
    assert_parent(learner, optimizer, parent)
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    fitting.step()
    if not all(torch.isfinite(p).all() for p in params):
        raise FloatingPointError('Nonfinite generator parameters')
    return {'pairs':records, 'mean_M_virtual_loss_before':sum(r['M_virtual_loss_before'] for r in records)/len(records),
        'gradient_norm':float(norm), 'parameter_update_norm':float((audit.vector(params, params)-before).norm())}


def summarize(output, c):
    landscapes = [read_json(output/f'state_{s}/landscape_pair_{i}.json')
                  for s in c['states'] for i in range(c['pairs'])]
    fits = [read_json(output/f'state_{s}/fit_{arm}.json') for s in c['states'] for arm in ARMS]
    for fit in fits:
        selected = [g for g in landscapes if g['state'] == fit['state']]
        rows = [r for g in selected for r in g['records']]
        rows += fit['initial']['pairs'] + fit['final']['pairs']
        parity = max(abs(r['virtual_real_parity'][kind]['M_virtual_minus_real'])
                     for r in rows for kind in ('same_gradient','meta_path'))
        repeat = max(abs(g['records'][1]['outer_M_loss']-g['records'][0]['outer_M_loss']) for g in selected)
        # A descriptive reference scale, not a confidence interval/error bound.
        roundoff = 8*torch.finfo(torch.float32).eps*max(1., max(abs(r['outer_M_loss']) for r in rows))
        reference = max(parity, repeat, roundoff)
        fit['numerical_reference'] = dict(max_abs_virtual_real_M_gap=parity,
            max_abs_uniform_repeat_M_gap=repeat, eight_epsilon_loss_scale=roundoff,
            reference_scale=reference, is_error_bound=False)
        fit['real_reduction_exceeds_reference'] = fit['real_loss_change'] < -reference
    return {'settings':c, 'landscapes':landscapes, 'fits':fits,
        'test_evaluated':False, 'development_evaluated':False, 'generalization_evidence':False}


def write_report(output, result):
    lines = ['# SYNAPSE fp32 fixed-set learnability check', '',
        'A/M reused for fitting and assessment. No held-out improvement claim.', '',
        'Frozen base weights are bf16-rounded and promoted to fp32; buffers are preserved. TF32 is off.', '',
        '| State | Arm | Virtual M change | Real M change | Numerical reference scale | Real reduction exceeds reference? |',
        '|---|---|---:|---:|---:|---|']
    for f in result['fits']:
        lines.append(f"| {f['state']} | {f['arm']} | {f['virtual_loss_change']:+.9g} | {f['real_loss_change']:+.9g} | {f['numerical_reference']['reference_scale']:.9g} | {f['real_reduction_exceeds_reference']} |")
    lines += ['', 'The reference is the largest measured virtual/real gap, uniform repeat drift, or eight fp32 epsilons times loss scale. It is a descriptive screen, not an error bound, significance test, or proof of learnability.', '',
        'Shared logits weight raw candidate slots (not hardness ranks). Pair logits have four free logits per A/M pair and cannot generalize by construction. The feature scorer is the original shared 4→32→1 network. All start uniform, receive equal full-set steps and the same outer optimizer settings; capacities differ.', '',
        '## Coarse mixture scan', '',
        '| State | Pair | Lowest-real-loss scanned recipe | Real M delta vs uniform |',
        '|---|---:|---|---:|']
    for g in result['landscapes']:
        rows = [r for r in g['records'] if r['recipe'] != 'uniform_repeat']
        best = min(rows, key=lambda r:r['outer_M_loss'])
        lines.append(f"| {g['state']} | {g['pair_index']} | {best['recipe']} | {best['outer_M_loss']-rows[0]['outer_M_loss']:+.9g} |")
    lines += ['', 'The scan includes uniform, four vertices, and six edge midpoints. Its minima are selected on M itself and are not validation results or a search over the whole simplex. Final fits use fixed steps, not best-step selection. Inspect per-pair gradients/weights and parity in learnability_summary.json before interpreting the means.', '',
        'If free pair weights improve but the feature scorer does not, investigate parameterization/optimization. If neither improves, that does not prove all generators or interventions ineffective. No automatic larger-model or full-benchmark launch occurs.']
    target = output/'LEARNABILITY_REPORT.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    os.replace(temporary, target)


def run(source, output, supplied, max_units=None):
    source, output = audit.safe_paths(source, output)
    c = settings(supplied)
    if max_units is not None and max_units < 1:
        raise ValueError('max-units must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    config, manifest, pairs, hashes, triple = audit.load_source(source, c)
    packages = ['torch','transformers','numpy']
    if config['backend'] == 'hf':
        packages += ['datasets','huggingface-hub','tokenizers','accelerate']
    identity = dict(version=1, settings=c, source_files_sha256=hashes,
        data_sha256=manifest['data_sha256'], pilot_source_sha256=source_hash(),
        diagnostic_source_sha256=file_hash(Path(__file__)), audit_source_sha256=file_hash(Path(audit.__file__)),
        versions={k:importlib.metadata.version(k) for k in packages},
        python=platform.python_version(), device=hardware(config), tf32=False, deterministic_algorithms=True,
        precision='bf16-rounded frozen parameters promoted to fp32; fp32 adapters/moments; original buffers',
        pairs=[{'A_id':i[0]['id'], 'M_id':o[0]['id']} for i,o in pairs])
    units = 0
    def boundary():
        if time.perf_counter()-started >= c['max_seconds'] or (max_units is not None and units >= max_units):
            raise audit.Pause('Budget reached at safe boundary; rerun with the same code/config/output to resume.')
    with run_lock(output):
        path = output/'learnability_manifest.json'
        if path.exists() and read_json(path)['identity'] != identity:
            raise ValueError('Learnability inputs/settings/environment changed; use a new output directory')
        if path.exists() and read_json(path)['status'] == 'complete':
            return read_json(path)
        atomic_json(path, dict(identity=identity, status='running', source_run=str(source),
            test_evaluated=False, development_evaluated=False, generalization_evidence=False))
        learner = optimizer = None
        try:
            boundary()
            tokenizer = load_tokenizer(config, manifest['resolved'])
            learner = load_learner(config, manifest['resolved'], triple[0])
            buffers = audit.buffer_fingerprint(learner)
            buffer_path = output/'model_buffers.json'
            if buffer_path.exists() and read_json(buffer_path) != buffers:
                raise ValueError('Loaded buffers differ from prior invocation')
            atomic_json(buffer_path, buffers)
            audit.set_precision(learner, 'bf16')
            audit.set_precision(learner, 'fp32')
            def check_buffers():
                if audit.buffer_fingerprint(learner) != buffers:
                    raise RuntimeError('Diagnostic changed model buffers')
            check_buffers()
            optimizer = make_optimizer(learner, config)
            for state in c['states']:
                parent = load_checkpoint(source/f"seed_{c['learner_seed']}/teacher/step_{state}.pt")['state']
                for index, pair in enumerate(pairs):
                    target = output/f'state_{state}/landscape_pair_{index}.json'
                    if target.exists():
                        continue
                    boundary()
                    restore(learner, optimizer, parent)
                    causality = audit.check_prompt_causality(learner, pair[0], tokenizer)
                    records = landscape(learner, optimizer, parent, pair, tokenizer, config)
                    check_buffers()
                    atomic_json(target, dict(state=state, pair_index=index, records=records, same_shape_causality=causality))
                    units += 1
                    print(f'State {state}, pair {index}: mixture scan saved', flush=True)
                for arm in ARMS:
                    target = output/f'state_{state}/fit_{arm}.pt'
                    report = target.with_suffix('.json')
                    if report.exists():
                        continue
                    generator = make_generator(arm, c, config['device'])
                    fitting = torch.optim.AdamW(generator.parameters(), lr=c['fit_lr'], eps=c['adam_eps'], weight_decay=0., foreach=False)
                    start, history = 0, []
                    if target.exists():
                        saved = load_checkpoint(target)
                        generator.load_state_dict(saved['generator'])
                        fitting.load_state_dict(saved['optimizer'])
                        start, history, initial = saved['step'], saved['history'], saved['initial']
                    else:
                        boundary()
                        initial = assess(learner, optimizer, parent, pairs, tokenizer, config, generator)
                        check_buffers()
                        atomic_torch(target, dict(step=0, generator=cpu(generator.state_dict()), optimizer=cpu(fitting.state_dict()), history=history, initial=initial))
                    for step in range(start, c['fit_steps']):
                        boundary()
                        row = fit_step(learner, optimizer, parent, pairs, tokenizer, config, generator, fitting)
                        check_buffers()
                        history.append(dict(step=step+1, **row))
                        atomic_torch(target, dict(step=step+1, generator=cpu(generator.state_dict()), optimizer=cpu(fitting.state_dict()), history=history, initial=initial))
                        units += 1
                        if (step+1) % 10 == 0 or step+1 == c['fit_steps']:
                            print(f'State {state}, {arm}: {step+1}/{c["fit_steps"]} full-set steps saved', flush=True)
                    boundary()
                    final = assess(learner, optimizer, parent, pairs, tokenizer, config, generator)
                    check_buffers()
                    atomic_json(report, dict(state=state, arm=arm, initial=initial, final=final, history=history,
                        parameter_count=sum(p.numel() for p in generator.parameters()),
                        real_loss_change=final['mean_M_real_loss']-initial['mean_M_real_loss'],
                        virtual_loss_change=final['mean_M_virtual_loss']-initial['mean_M_virtual_loss'],
                        is_generalization_result=False))
                    units += 1
                    del generator, fitting
            if any(file_hash(source/name) != h for name,h in hashes.items()):
                raise ValueError('Source files changed during diagnostic')
            check_buffers()
            result = summarize(output, c)
            atomic_json(output/'learnability_summary.json', result)
            write_report(output, result)
            status = 'complete'
        except audit.Pause as exc:
            status = 'paused'
            print(str(exc), flush=True)
        except Exception as exc:
            atomic_json(path, {**read_json(path), 'status':'failed', 'error':f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            del learner, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_json(path, {**read_json(path), 'status':status, 'last_invocation_seconds':time.perf_counter()-started})
        return read_json(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','config'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--max-units', type=int)
    args = parser.parse_args()
    print(run(args.source, args.output, read_json(args.config), args.max_units)['status'])


if __name__ == '__main__':
    main()
