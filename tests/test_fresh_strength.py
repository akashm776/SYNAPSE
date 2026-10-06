from pathlib import Path

import pytest
import torch

from synapse import audit, fresh_strength as fresh
from synapse.llm.experiment import execute, file_hash, read_json


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp('fresh_strength_source')
    config=read_json(Path(__file__).parents[1]/'configs/llm_smoke.json')
    execute('preflight',config,root)
    execute('run',config,root)
    return root


def test_settings_and_outlier_statistics():
    for supplied in ({'pairs':33},{'pairs':0},{'selection_seed':123},{'alphas':[.1,1.]},{'alphas':[0,2]}):
        with pytest.raises(ValueError):
            fresh.settings(supplied)
    assert len(fresh.recipes())==5
    d=fresh.describe([-10.,1.,1.],[-10.,1.,1.],[.1]*3)
    assert d['mean_real']==pytest.approx(-8/3)
    assert d['median_real']==1
    assert d['helpful_pairs']==1 and d['harmful_pairs']==2
    assert d['leave_one_pair_out_mean_range']==[-4.5,1.]
    assert d['reductions_exceeding_reference']==1
    assert d['increases_exceeding_reference']==2
    assert fresh.describe([0.],[0.],[.1])['leave_one_pair_out_mean_range'] is None


def test_fresh_selection_excludes_both_old_components(source):
    c=fresh.settings({'states':[2],'pairs':2,'alphas':[0,.05]})
    config,manifest,pairs,hashes,triple,excluded=fresh.load_source(source,c)
    old=audit.load_source(source,{**c,'pairs':2,'selection_seed':61005})[2]
    assert excluded==old
    old_ids={r[0]['id'] for pair in old for r in pair}
    fresh_ids=[r[0]['id'] for pair in pairs for r in pair]
    assert len(set(fresh_ids))==4 and not old_ids.intersection(fresh_ids)
    assert fresh.load_source(source,c)[2]==pairs
    # M has six eligible examples in this fixture: never wrap/reuse an epoch.
    with pytest.raises(ValueError,match='Too few'):
        fresh.load_source(source,{**c,'pairs':5})


def test_fresh_resume_scope_and_aggregate(source,tmp_path,monkeypatch):
    from synapse.llm import model
    c=read_json(Path(__file__).parents[1]/'configs/fresh_strength_smoke.json')
    source_hashes={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    pairs=fresh.load_source(source,c)[2]
    allowed={r[0]['id'] for pair in pairs for r in pair}
    original=model.objective
    def guarded(learner,rows,*args,**kwargs):
        assert all(r['id'] in allowed for r in rows)
        assert all(p.dtype==torch.float32 for p in learner.parameters())
        return original(learner,rows,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError('No evaluation or completed-run model reload')
    monkeypatch.setattr(model,'objective',guarded)
    monkeypatch.setattr(audit,'objective',guarded)
    monkeypatch.setattr(model,'evaluate',forbidden)
    resumed,full=tmp_path/'resumed',tmp_path/'full'
    assert fresh.run(source,resumed,c,max_units=1)['status']=='paused'
    first=resumed/'state_2/pair_0_alpha_0.json'
    h=file_hash(first)
    assert fresh.run(source,resumed,c)['status']=='complete'
    assert file_hash(first)==h
    assert fresh.run(source,full,c)['status']=='complete'
    a,b=[read_json(p/'fresh_strength_summary.json') for p in (resumed,full)]
    for s in (a,b):
        for g in s['groups']:
            g.pop('seconds')
    assert a==b
    assert len(a['groups'])==4 and len(a['summaries'])==2
    assert a['selected_alpha'] is None and not a['generalization_evidence']
    for g in a['groups']:
        assert len(g['records'])==8 and g['buffers_unchanged']
        for r in g['records']:
            if 'vs_first_repeat' in r:
                assert r['M_real_repeat_difference']==0
                assert all(v['bitwise_equal'] for v in r['vs_first_repeat'].values())
            if g['alpha']==0:
                assert r['M_real_delta_vs_native']==0
                assert r['vs_native']['update']['bitwise_equal']
    for summary in a['summaries']:
        groups=[g for g in a['groups'] if g['alpha']==summary['alpha']]
        for name,_ in fresh.recipes():
            values=[r['M_real_delta_vs_native'] for g in groups for r in g['records'] if r['recipe']==name]
            assert summary['contrasts'][name+'_vs_native']['mean_real']==sum(values)/2
        for i,g in enumerate(groups):
            rows={r['recipe']:r for r in g['records']}
            expected=rows['uniform']['outer_M_loss']-sum(rows[f'vertex_{j}']['outer_M_loss'] for j in range(4))/4
            assert summary['contrasts']['uniform_vs_average_single']['per_pair_real'][i]==expected
    assert source_hashes=={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    manifest=read_json(resumed/'fresh_strength_manifest.json')
    assert len(manifest['identity']['excluded_pairs'])==2
    assert len(manifest['identity']['pairs'])==2
    with pytest.raises(ValueError,match='changed'):
        fresh.run(source,resumed,{**c,'alphas':[0,.05,1.]})
    monkeypatch.setattr(fresh,'load_learner',forbidden)
    assert fresh.run(source,resumed,c)['status']=='complete'
    assert fresh.run(source,tmp_path/'budget',{**c,'max_seconds':1e-9})['status']=='paused'
    with pytest.raises(ValueError):
        fresh.run(source,source/'nested',c)
    with pytest.raises(ValueError,match='original source strength'):
        fresh.run(source,tmp_path/'bad_alpha',{**c,'alphas':[0,.01]})
