from copy import deepcopy
from pathlib import Path

import pytest
import torch

from synapse import audit
from synapse.llm.experiment import execute, file_hash, read_json


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp('audit_source')
    config=read_json(Path(__file__).parents[1]/'configs/llm_smoke.json')
    execute('preflight',config,root)
    execute('run',config,root)
    return root


def test_audit_settings_and_output_safety(tmp_path):
    for values in ({'repeats':1},{'epsilons':[.25]},{'max_seconds':float('nan')}, {'typo':1}):
        with pytest.raises(ValueError):
            audit.settings(values)
    for output in (tmp_path,tmp_path/'nested',tmp_path.parent):
        with pytest.raises(ValueError):
            audit.safe_paths(tmp_path,output)
    assert audit.safe_paths(tmp_path,tmp_path.parent/'separate')[0]==tmp_path.resolve()


def test_fixed_probe_matches_uniform_and_live_gradients():
    from synapse.llm.core import auxiliary
    torch.manual_seed(4)
    q,p,n=[torch.randn(*shape,requires_grad=True) for shape in ((2,8),(2,8),(2,4,8))]
    ref,_,_=auxiliary(q,p,n,'uniform_barycenter')
    actual,_,weights=auxiliary(q,p,n,'learned_barycenter',audit.Probe())
    assert torch.equal(ref,actual)
    assert torch.equal(weights,torch.full_like(weights,.25))
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(actual,(q,p,n)))
    w=audit.Probe(epsilon=.001)(torch.zeros(1,4,4))
    torch.testing.assert_close(w,torch.tensor([[.251,.249,.25,.25]]))


def test_precision_preserves_adapters_and_rounded_weights():
    from synapse.llm.model import load_learner
    from synapse.llm.experiment import validate_config
    config=validate_config(read_json(Path(__file__).parents[1]/'configs/llm_smoke.json'))
    learner=load_learner(config,{},101)
    trainable={k:p.detach().clone() for k,p in learner.trainable().items()}
    buffers={k:b.clone() for k,b in learner.named_buffers()}
    fingerprint=audit.buffer_fingerprint(learner)
    assert buffers and learner.backbone.model.rotary_emb.inv_freq.dtype==torch.float32
    # Ensure this fixture actually detects the v1 lossy buffer cast.
    freq=learner.backbone.model.rotary_emb.inv_freq
    assert not torch.equal(freq,freq.bfloat16().float())
    audit.set_precision(learner,'bf16')
    rounded={k:p.float().detach().clone() for k,p in learner.named_parameters() if not p.requires_grad}
    audit.set_precision(learner,'fp32')
    assert all(torch.equal(p,rounded[k]) for k,p in learner.named_parameters() if not p.requires_grad)
    assert all(p.dtype==torch.float32 and torch.equal(p,trainable[k]) for k,p in learner.trainable().items())
    for precision in ('bf16','fp32','bf16'):
        audit.set_precision(learner,precision)
        assert audit.buffer_fingerprint(learner)==fingerprint
        assert all(b.dtype==buffers[k].dtype and torch.equal(b,buffers[k]) for k,b in learner.named_buffers())


def test_difference_uses_stable_bounded_cosines():
    x=torch.linspace(-.01,.01,100001)
    assert audit.difference(x,x)['bitwise_equal']
    assert audit.difference(x,x)['cosine']==pytest.approx(1.)
    assert audit.difference(x,-x)['cosine']==pytest.approx(-1.)
    assert audit.difference(x,torch.zeros_like(x))['cosine'] is None


def test_audit_resume_source_immutability_and_no_reporting(source,tmp_path,monkeypatch):
    from synapse.llm import model
    config=read_json(Path(__file__).parents[1]/'configs/audit_smoke.json')
    config['max_seconds']=300
    initial={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    data=read_json(source/'data.json')
    allowed={r['id'] for k in ('A','M') for r in data['partitions'][k]}
    original=model.objective
    def guarded(learner,rows,*args,**kwargs):
        assert all(r['id'] in allowed for r in rows)
        return original(learner,rows,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError('Audit must never call decoding/evaluation')
    monkeypatch.setattr(model,'objective',guarded)
    monkeypatch.setattr(audit,'objective',guarded)
    monkeypatch.setattr(model,'evaluate',forbidden)
    resumed=tmp_path/'resumed'
    assert audit.run_audit(source,resumed,config,max_units=2)['status']=='paused'
    group=resumed/'state_2/repeatability_pair_0.json'
    first_hash=file_hash(group)
    assert audit.run_audit(source,resumed,config)['status']=='complete'
    assert file_hash(group)==first_hash
    uninterrupted=tmp_path/'uninterrupted'
    assert audit.run_audit(source,uninterrupted,config)['status']=='complete'
    a,b=[read_json(p/'audit_summary.json') for p in (resumed,uninterrupted)]
    for result in (a,b):
        for g in result['repeatability']:
            g.pop('seconds')
    assert a==b
    assert not a['test_evaluated'] and not a['generalization_evidence']
    assert len(a['fits'])==2
    assert read_json(resumed/'audit_manifest.json')['identity']['audit_version']==2
    assert read_json(resumed/'model_buffers.json')==a['repeatability'][0]['buffers']
    assert a['repeatability'][0]['buffers_unchanged']
    rows=[r for r in a['repeatability'][0]['records'] if 'condition' in r]
    assert all(set(r['virtual_real_parity'])=={'same_gradient','meta_path'} for r in rows)
    for r in rows:
        for check in r['virtual_real_parity'].values():
            assert check['M_virtual_minus_real']==check['virtual_M_loss']-r['outer_M_loss']
            if r['precision']=='fp32':
                assert check['update']['max_abs'] < 1e-6
                assert abs(check['M_virtual_minus_real']) < 1e-5
    assert all(r['vs_first_repeat']['update']['bitwise_equal'] for r in a['repeatability'][0]['records'] if 'vs_first_repeat' in r)
    assert initial=={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    assert (resumed/'AUDIT_REPORT.md').exists()
    with pytest.raises(ValueError,match='changed'):
        audit.run_audit(source,resumed,{**config,'fit_lr':.002})
    # No implicit preparation/rewrite of a source with modified data.
    from synapse.llm.experiment import atomic_json
    original_data=deepcopy(data)
    data['partitions']['A'][0]['answer']='corrupted'
    atomic_json(source/'data.json',data)
    try:
        with pytest.raises(ValueError,match='hash mismatch'):
            audit.run_audit(source,tmp_path/'bad',config)
    finally:
        atomic_json(source/'data.json',original_data)


def test_wall_budget_pauses_before_model_load(source,tmp_path,monkeypatch):
    config=read_json(Path(__file__).parents[1]/'configs/audit_smoke.json')
    config['max_seconds']=1e-9
    def forbidden(*args,**kwargs):
        raise AssertionError('Expired budget must not load a model')
    monkeypatch.setattr(audit,'load_learner',forbidden)
    result=audit.run_audit(source,tmp_path/'paused',config)
    assert result['status']=='paused'
    assert not (tmp_path/'paused/audit_summary.json').exists()
