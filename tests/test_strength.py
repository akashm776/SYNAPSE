from pathlib import Path

import pytest
import torch

from synapse import audit, strength
from synapse.llm.experiment import execute, file_hash, read_json


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp('strength_source')
    config=read_json(Path(__file__).parents[1]/'configs/llm_smoke.json')
    execute('preflight',config,root)
    execute('run',config,root)
    return root


def test_strength_settings_and_recipes():
    for values in ({'alphas':[.01]}, {'alphas':[0,0,.01]}, {'alphas':[0,2]},
                   {'alphas':[0,float('nan')]}, {'alphas':[0,.1,.01]},
                   {'alphas':[0,True]}, {'pairs':0}, {'typo':1}):
        with pytest.raises(ValueError):
            strength.settings(values)
    recipes=strength.recipes()
    assert len(recipes)==11 and len({n for n,w in recipes})==11
    assert all(len(w)==4 and sum(w)==1 and min(w)>=0 for n,w in recipes)
    probe=strength.StrengthProbe()
    assert probe.diagnostics()=={}
    probe(torch.zeros(1,4,4))
    assert probe.diagnostics()['weights']==[[.25]*4]


def test_strength_resume_controls_and_scope(source,tmp_path,monkeypatch):
    from synapse.llm import model
    c=read_json(Path(__file__).parents[1]/'configs/strength_smoke.json')
    c['alphas']=[0,.05,.5]
    c['pairs']=2
    original_files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    data=read_json(source/'data.json')
    allowed={r['id'] for k in ('A','M') for r in data['partitions'][k]}
    original=model.objective
    zero_learned_calls=[]
    def guarded(learner,rows,tokenizer,config,arm='native',*args,**kwargs):
        assert all(r['id'] in allowed for r in rows)
        assert all(p.dtype==torch.float32 for p in learner.parameters())
        if config['alpha']==0 and arm=='learned_barycenter':
            zero_learned_calls.append(1)
        return original(learner,rows,tokenizer,config,arm,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError('No reporting in strength diagnostic')
    monkeypatch.setattr(model,'objective',guarded)
    monkeypatch.setattr(audit,'objective',guarded)
    monkeypatch.setattr(model,'evaluate',forbidden)
    resumed,full=tmp_path/'resumed',tmp_path/'full'
    assert strength.run(source,resumed,c,max_units=1)['status']=='paused'
    saved=resumed/'state_2/pair_0_alpha_0.json'
    saved_hash=file_hash(saved)
    assert strength.run(source,resumed,c)['status']=='complete'
    assert file_hash(saved)==saved_hash
    assert strength.run(source,full,c)['status']=='complete'
    a,b=[read_json(p/'strength_summary.json') for p in (resumed,full)]
    for s in (a,b):
        for g in s['groups']:
            g.pop('seconds')
    assert a==b and zero_learned_calls
    assert len(a['groups'])==6 and len(a['summaries'])==3
    assert a['selected_alpha'] is None and a['selected_recipe'] is None
    assert not a['test_evaluated'] and not a['generalization_evidence']
    for g in a['groups']:
        assert len(g['records'])==14 and g['buffers_unchanged']
        native=g['records'][0]
        uniform=g['records'][2]
        for r in g['records']:
            assert r['M_real_delta_vs_native']==r['outer_M_loss']-native['outer_M_loss']
            assert 0 < r['clip_factor'] <= 1
            if r['specified_weights'] is not None:
                assert r['M_real_delta_vs_uniform']==r['outer_M_loss']-uniform['outer_M_loss']
            if 'vs_first_repeat' in r:
                assert r['M_real_repeat_difference']==0
                assert all(v['bitwise_equal'] for v in r['vs_first_repeat'].values())
            if g['alpha']==0:
                assert not r['auxiliary_active'] and 'weights' not in r
                assert r['M_real_delta_vs_native']==r['M_virtual_delta_vs_native']==0
                assert r['vs_native']['update']['bitwise_equal']
                assert r['weighted_aux_gradient_norm']==0
    small,big=a['groups'][1:3]
    for x,y in zip(small['records'][2:-1],big['records'][2:-1]):
        # Unclipped auxiliary gradient scales with alpha; clipping/Adam need not.
        assert y['weighted_aux_gradient_norm']==pytest.approx(10*x['weighted_aux_gradient_norm'],rel=1e-4,abs=1e-6)
    for summary in a['summaries']:
        groups=[g for g in a['groups'] if g['alpha']==summary['alpha']]
        for row in summary['recipes']:
            values=[r['M_real_delta_vs_native'] for g in groups for r in g['records'] if r['recipe']==row['recipe']]
            assert row['per_pair_real_delta_vs_native']==values
            assert row['mean_real_delta_vs_native']==sum(values)/2
        assert not summary['numerical_reference']['is_error_bound']
    assert original_files=={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    assert (resumed/'STRENGTH_REPORT.md').exists()
    with pytest.raises(ValueError,match='changed'):
        strength.run(source,resumed,{**c,'selection_seed':999})
    monkeypatch.setattr(strength,'load_learner',forbidden)
    assert strength.run(source,resumed,c)['status']=='complete'


def test_strength_budget_and_source_alpha_guard(source,tmp_path,monkeypatch):
    c=read_json(Path(__file__).parents[1]/'configs/strength_smoke.json')
    def forbidden(*args,**kwargs):
        raise AssertionError('Must stop before model load')
    monkeypatch.setattr(strength,'load_learner',forbidden)
    assert strength.run(source,tmp_path/'paused',{**c,'max_seconds':1e-9})['status']=='paused'
    with pytest.raises(ValueError,match='original source strength'):
        strength.run(source,tmp_path/'wrong',{**c,'alphas':[0,.01]})
    with pytest.raises(ValueError):
        strength.run(source,source/'child',c)
