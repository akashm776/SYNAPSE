from pathlib import Path

import pytest
import torch

from synapse import audit, learnability as diagnostic
from synapse.llm.experiment import execute, file_hash, read_json


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp('learnability_source')
    config = read_json(Path(__file__).parents[1]/'configs/llm_smoke.json')
    execute('preflight', config, root)
    execute('run', config, root)
    return root


def test_settings_and_direct_candidate_slots():
    for values in ({'adam_eps':0}, {'adam_eps':float('nan')}, {'pairs':0},
                   {'fit_steps':0}, {'typo':1}, {'fit_lr':-1}):
        with pytest.raises(ValueError):
            diagnostic.settings(values)
    features = torch.randn(1,4,4)
    generator = diagnostic.DirectWeights(2)
    assert torch.equal(generator(features), torch.full((1,4),.25))
    with torch.no_grad():
        generator.logits[1,0] = 2
    diagnostic.use_pair(generator, 1)
    assert generator(features)[0,0] > .5
    assert torch.equal(generator(features), generator(features.flip(1)))
    generator(features)[0,0].backward()
    assert torch.count_nonzero(generator.logits.grad[0]) == 0
    assert torch.count_nonzero(generator.logits.grad[1]) == 4
    diagnostic.use_pair(generator, 0)
    assert torch.equal(generator(features), torch.full((1,4),.25))


def test_direct_logits_live_meta_derivative():
    from synapse.llm.core import auxiliary
    torch.manual_seed(55)
    q,p,n = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
             for shape in ((1,5),(1,5),(1,4,5))]
    gen = diagnostic.DirectWeights(1).double()
    # Derivative through an inner gradient must reach logits, not only the loss.
    loss,_,_ = auxiliary(q,p,n,'learned_barycenter',gen)
    inner = torch.autograd.grad(loss,q,create_graph=True)[0]
    grad = torch.autograd.grad(inner.square().sum(), gen.logits)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 1e-8
    assert abs(float(grad.sum())) < 1e-10
    def value():
        loss,_,_ = auxiliary(q,p,n,'learned_barycenter',gen)
        return float(torch.autograd.grad(loss,q)[0].square().sum())
    h = 1e-5
    for index in range(4):
        with torch.no_grad():
            gen.logits[0,index] = h
        plus = value()
        with torch.no_grad():
            gen.logits[0,index] = -h
        minus = value()
        with torch.no_grad():
            gen.logits[0,index] = 0
        assert float(grad[0,index]) == pytest.approx((plus-minus)/(2*h), rel=1e-5, abs=1e-7)


def test_fit_step_averages_pairs_before_optimizer_update(monkeypatch):
    monkeypatch.setattr(diagnostic, 'restore', lambda *a:None)
    monkeypatch.setattr(diagnostic, 'assert_parent', lambda *a:None)
    def meta(learner, optimizer, inner, outer, tokenizer, config, arm, probe):
        weights = probe(torch.zeros(1,4,4))
        return (weights-torch.tensor(inner)).square().sum(), {}
    monkeypatch.setattr(diagnostic, 'meta_objective', meta)
    c = diagnostic.settings({'pairs':2})
    pairs = [([1.,0.,0.,0.],None), ([0.,0.,1.,0.],None)]
    for arm in diagnostic.ARMS:
        actual = diagnostic.make_generator(arm,c,'cpu')
        expected = diagnostic.make_generator(arm,c,'cpu')
        opt = torch.optim.AdamW(actual.parameters(),lr=.01,weight_decay=0.,foreach=False)
        ref = torch.optim.AdamW(expected.parameters(),lr=.01,weight_decay=0.,foreach=False)
        loss = 0
        for i,(target,_) in enumerate(pairs):
            diagnostic.use_pair(expected,i)
            loss = loss + (expected(torch.zeros(1,4,4))-torch.tensor(target)).square().sum()/2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
        ref.step()
        row = diagnostic.fit_step(None,None,None,pairs,None,{},actual,opt)
        assert len(row['pairs']) == 2
        for p,q in zip(actual.parameters(),expected.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_full_set_resume_and_source_safety(source, tmp_path, monkeypatch):
    from synapse.llm import model
    config = read_json(Path(__file__).parents[1]/'configs/learnability_smoke.json')
    before = {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    data = read_json(source/'data.json')
    allowed = {r['id'] for key in ('A','M') for r in data['partitions'][key]}
    original = model.objective
    def guarded(learner, rows, *args, **kwargs):
        assert all(r['id'] in allowed for r in rows)
        assert all(p.dtype == torch.float32 for p in learner.parameters())
        return original(learner, rows, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('No reporting allowed')
    monkeypatch.setattr(model, 'objective', guarded)
    monkeypatch.setattr(audit, 'objective', guarded)
    monkeypatch.setattr(model, 'evaluate', forbidden)
    resumed, full = tmp_path/'resumed', tmp_path/'full'
    # Two scans then one optimizer step: resume within the first fit.
    assert diagnostic.run(source, resumed, config, max_units=3)['status'] == 'paused'
    checkpoint = torch.load(resumed/'state_2/fit_shared_logits.pt', weights_only=True)
    assert checkpoint['step'] == 1
    scan = resumed/'state_2/landscape_pair_0.json'
    saved_scan = file_hash(scan)
    assert diagnostic.run(source, resumed, config)['status'] == 'complete'
    assert file_hash(scan) == saved_scan
    assert diagnostic.run(source, full, config)['status'] == 'complete'
    a,b = [read_json(p/'learnability_summary.json') for p in (resumed, full)]
    assert a == b
    assert not a['test_evaluated'] and not a['generalization_evidence']
    assert len(a['fits']) == 3 and len(a['landscapes']) == 2
    for g in a['landscapes']:
        assert len(g['records']) == 12
        assert g['records'][0]['outer_M_loss'] == g['records'][1]['outer_M_loss']
    for f in a['fits']:
        assert len(f['history']) == 2
        assert all(len(r['pairs']) == 2 for r in f['history'])
        assert f['real_loss_change'] == f['final']['mean_M_real_loss']-f['initial']['mean_M_real_loss']
        assert not f['numerical_reference']['is_error_bound']
        for r in f['initial']['pairs'] + f['final']['pairs']:
            assert abs(r['virtual_real_parity']['meta_path']['M_virtual_minus_real']) < 1e-5
    assert a['fits'][0]['parameter_count'] == 4
    assert a['fits'][1]['parameter_count'] == 8
    assert a['fits'][2]['parameter_count'] == 193
    assert before == {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    assert (resumed/'LEARNABILITY_REPORT.md').exists()
    with pytest.raises(ValueError, match='changed'):
        diagnostic.run(source, resumed, {**config,'fit_lr':.02})
    def no_load(*args, **kwargs):
        raise AssertionError('Complete resume must not load model')
    monkeypatch.setattr(diagnostic, 'load_learner', no_load)
    assert diagnostic.run(source, resumed, config)['status'] == 'complete'


def test_budget_and_path_guards(source, tmp_path, monkeypatch):
    config = read_json(Path(__file__).parents[1]/'configs/learnability_smoke.json')
    def no_load(*args, **kwargs):
        raise AssertionError('Expired budget must not load model')
    monkeypatch.setattr(diagnostic, 'load_learner', no_load)
    assert diagnostic.run(source, tmp_path/'paused', {**config,'max_seconds':1e-9})['status'] == 'paused'
    with pytest.raises(ValueError):
        diagnostic.run(source, source/'child', config)
