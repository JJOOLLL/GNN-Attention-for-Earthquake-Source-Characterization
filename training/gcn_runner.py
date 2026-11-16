"""Portable final GCN training: seed42, SiLU plateau, then ReLU fine-tuning.

The numerical loop is preserved from the completed experiment. Public names,
paths, source loading and optional reference-environment auditing are adapted.
Use the repository run.py entry point; importing this module never trains.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
import time
import warnings

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MODELS = ('gcn_max', 'gcn_attention')
REGIONS = ('California', 'Alaska')
TARGETS = tuple(f'feature_extractor.{i}.1' for i in (0,1,2,4,5,6,8,9,10,12,13))
POLICY = dict(silu_min_epochs=150, silu_max_epochs=1000,
              relu_min_epochs=50, relu_max_epochs=200,
              silu_lr=1e-3, relu_lr=2e-5, seed=42,
              stability='consecutive validation-loss non-improvements; original patience 50 or 70',
              switch='restore best SiLU weights, replace encoder activations, fresh Adam',
              selection='original validation weighted L1; independent ReLU best including epoch zero',
              probe='first 8 validation events only; preserve all training RNG states')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def preflight(model, region, graph_scheme='knn', strict_reference=False):
    from training.portable_sources import load_sources
    cells, meta = load_sources(model, region, graph_scheme)
    meta.update(encoder_activation='SiLU_then_ReLU_first_11',
                initial_checkpoint=None, training_from_scratch=True,
                switch_policy=POLICY, strict_reference=bool(strict_reference),
                runner_sha256=sha(__file__),
                checkpointing_sha256=sha(ROOT/'training'/'checkpointing.py'),
                portable_sources_sha256=sha(ROOT/'training'/'portable_sources.py'),
                metrics_sha256=sha(ROOT/'training'/'metrics.py'))
    return cells, meta


def set_encoder(model, activation):
    """Replace only parameter-free activations, without drawing random numbers."""
    import torch.nn as nn
    require(activation in ('silu','relu'), 'Bad activation')
    before = {k:type(v) for k,v in model.named_modules()}
    for name in TARGETS:
        i = int(name.split('.')[1])
        old = model.feature_extractor[i][1]
        require(type(old) in (nn.ReLU,nn.SiLU), 'Unexpected encoder module: '+name)
        model.feature_extractor[i][1] = (nn.SiLU() if activation=='silu' else nn.ReLU()).train(old.training)
    after = {k:type(v) for k,v in model.named_modules()}
    require(before.keys()==after.keys(), 'Module structure changed')
    require(all(before[k] is after[k] for k in before if k not in TARGETS), 'Non-encoder module changed')
    expected = nn.SiLU if activation=='silu' else nn.ReLU
    require(all(after[k] is expected for k in TARGETS), 'Incomplete activation replacement')
    require(isinstance(model.feature_extractor[14][1],nn.Tanh), 'Last CNN Tanh must remain unchanged')
    require(not model.use_bn, 'Unexpected BatchNorm configuration')
    return list(TARGETS)


@contextmanager
def preserve_rng_and_mode(model):
    from training.checkpointing import capture_rng, restore_rng
    rng = capture_rng()
    modes = [(m,m.training) for m in model.modules()]
    try:
        yield
    finally:
        for module,mode in modes:
            module.training = mode
        restore_rng(rng)


def validation(model, loader, device, loss_weights):
    import torch
    model.eval()
    total = 0.
    with torch.no_grad():
        for (waves,coords,mask), labels, _ in loader:
            waves,coords,mask,labels = (v.to(device) for v in (waves,coords,mask,labels))
            p = model(waves,coords,mask)
            total += float((torch.abs(p-labels)*loss_weights).mean().item())
    value = total / len(loader)
    require(math.isfinite(value), 'Nonfinite validation loss')
    return value


def physical(x, ns):
    import numpy as np
    return np.stack([(x[:,0]/2+.5)*(ns['maxlatitude']-ns['minlatitude'])+ns['minlatitude'],
                     (x[:,1]/2+.5)*(ns['maxlongitude']-ns['minlongitude'])+ns['minlongitude'],
                     (x[:,2]/2+.5)*ns['maxdepth']/1000,
                     (x[:,3]/2+.5)*(ns['maxmag']-ns['minmag'])+ns['minmag']],axis=1)


def make_probe(ns):
    import torch
    from torch.utils.data._utils.collate import default_collate
    if hasattr(ns['model'], '_graph_builder'):
        prepare_graphs(ns)
    with preserve_rng_and_mode(ns['model']):
        items = [ns['val_dataset'][i] for i in range(min(8,len(ns['val_dataset'])))]
        (x,c,m),y,_ = default_collate(items)
    return tuple(t.to(ns['device']) for t in (x,c,m,y))


def prepare_graphs(ns):
    """Original RNG-free exact-graph precomputation, before the first probe."""
    from training.checkpointing import capture_rng, rng_digest, state_digest
    model = ns['model']
    builder = model._graph_builder
    before_rng, before_weights = rng_digest(capture_rng()), state_digest(model)
    out = Path(ns['RUN_DIR'])
    start = time.perf_counter()
    record = {'state': 'precomputing', 'started_utc': now(), 'completed_splits': []}

    def progress(*args, **kwargs):
        record.update(updated_utc=now(), seconds=time.perf_counter() - start,
                      progress_detail=args[0] if len(args) == 1 and isinstance(args[0], dict) else kwargs)
        write_json(out / 'graph_precompute.json', record)
        if time.perf_counter() - progress.last_print >= 20:
            print('GRAPH_PRECOMPUTE=' + json.dumps(record), flush=True)
            progress.last_print = time.perf_counter()

    progress.last_print = start - 20
    write_json(out / 'graph_precompute.json', record)
    for split in ('train', 'val', 'test'):
        record['current_split'] = split
        write_json(out / 'graph_precompute.json', record)
        summary = builder.precompute_dataset(ns[split + '_dataset'], split, progress_callback=progress)
        record['completed_splits'].append({'split': split, 'summary': summary})
    require(before_rng == rng_digest(capture_rng()), 'Graph precompute consumed training RNG')
    require(before_weights == state_digest(model), 'Graph precompute changed model weights')
    record.update(state='completed', completed_utc=now(), seconds=time.perf_counter() - start,
                  rng_unchanged=True, parameters_unchanged=True)
    write_json(out / 'graph_precompute.json', record)
    print('GRAPH_PRECOMPUTE_COMPLETED=' + json.dumps(record), flush=True)


def probe(model, batch, ns, region):
    """Small fixed validation probe. No gradients or optimizer updates."""
    import torch
    from training.checkpointing import capture_rng,rng_digest,state_digest
    from training.metrics import metrics
    x,c,mask,y = batch
    before_rng = rng_digest(capture_rng())
    before_weights = state_digest(model)
    features=[]
    layers={}
    hooks=[]
    collect_layers=True
    def capture_layer(name):
        def hook(_m,_args,out):
            if not collect_layers:
                return
            valid=out.permute(0,2,1,3)[mask.bool()]
            active=(valid>0).any(dim=0).any(dim=-1)
            layers[name]={'positive_channels':int(active.sum()), 'channels':len(active),
                          'zero_fraction':float((valid==0).float().mean()),
                          'mean_abs':float(valid.abs().mean())}
        return hook
    with preserve_rng_and_mode(model):
        model.eval()
        for name in TARGETS:
            hooks.append(model.get_submodule(name).register_forward_hook(capture_layer(name)))
        hooks.append(model.feature_extractor.register_forward_hook(
            lambda _m,_args,out:features.append(out.detach().amax(dim=3).permute(0,2,1).clone())))
        try:
            with torch.inference_mode():
                normal=model(x,c,mask)
                collect_layers=False
                zero=model(torch.zeros_like(x),c,mask)
                diff=(features[0]-features[1]).abs()[mask.bool()]
                report=dict(events=len(x), valid_station_vectors=len(diff),
                    changed_station_vectors=int(diff.any(dim=1).sum()),
                    encoder_max_abs_wave_vs_zero_diff=float(diff.max()),
                    changed_epicentres=int((normal[:,:2]!=zero[:,:2]).any(dim=1).sum()),
                    layers=layers,
                    normal=metrics(physical(y.cpu().numpy(),ns),physical(normal.cpu().numpy(),ns),region),
                    zero_waveform=metrics(physical(y.cpu().numpy(),ns),physical(zero.cpu().numpy(),ns),region))
        finally:
            for h in hooks:
                h.remove()
    require(before_rng==rng_digest(capture_rng()), 'Diagnostic consumed training RNG')
    require(before_weights==state_digest(model), 'Diagnostic changed model weights')
    return report


def train_epoch(model, optimizer, loader, device, loss_weights):
    import torch
    model.train()
    total=0.
    grad=None
    for batch_i,((waves,coords,mask),labels,_) in enumerate(loader):
        waves,coords,mask,labels=(x.to(device) for x in (waves,coords,mask,labels))
        optimizer.zero_grad()
        p=model(waves,coords,mask)
        loss=(torch.abs(p-labels)*loss_weights).mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        if batch_i==0:
            grad={}
            for i in (0,13):
                g=model.feature_extractor[i][0].weight.grad
                grad[str(i)]={'present':g is not None,
                    'norm':float(g.norm()) if g is not None else None,
                    'nonzero':int(torch.count_nonzero(g)) if g is not None else 0}
        optimizer.step()
        total+=loss.item()
    return total/len(loader),grad


def save_full(path, model, optimizer, state, best, history, meta):
    from training.checkpointing import atomic_torch_save,capture_rng,_cpu_copy,_check_finite_tensors
    payload=dict(format_version=1,meta=meta,model=_cpu_copy(model.state_dict()),
                 optimizer=_cpu_copy(optimizer.state_dict()),state=state.copy(),
                 best=_cpu_copy(best),history=history,rng=capture_rng())
    _check_finite_tensors(payload,'checkpoint')
    atomic_torch_save(payload,path,rotate=True)


def restore_full(path, model, optimizer, meta):
    import torch
    from training.checkpointing import restore_rng,_check_finite_tensors
    payload=None
    errors=[]
    for p in (path,path.with_suffix('.previous.pth')):
        try:
            candidate=torch.load(p,map_location='cpu',weights_only=True)
            _check_finite_tensors(candidate,'checkpoint')
        except Exception as e:
            errors.append(str(e))
            continue
        require(candidate.get('format_version')==1 and candidate.get('meta')==meta,'Checkpoint provenance mismatch')
        payload=candidate
        break
    require(payload is not None,'No valid resume checkpoint: '+str(errors))
    set_encoder(model,payload['state']['stage'])
    model.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    return payload['state'],payload['best'],payload['history']


def train(name,region,out,resume=False,graph_scheme='knn',strict_reference=False):
    import numpy as np
    import psutil
    import torch
    from training.checkpointing import atomic_torch_save,capture_rng,rng_digest,state_digest,_cpu_copy
    from training import portable_sources as checker
    from training.metrics import metrics
    from experiments.paths import get_save_root
    cells,meta=preflight(name,region,graph_scheme,strict_reference)
    require(torch.cuda.is_available(),'CUDA unavailable; no silent CPU fallback')
    require(psutil.virtual_memory().available>40*2**30,'Less than 40 GiB RAM available')
    out=Path(out).resolve()
    expected=(get_save_root()/region/name/graph_scheme).resolve()
    require(out==expected,'Output differs from the selected model/region/graph directory')
    disk_parent=out
    while not disk_parent.exists():
        disk_parent=disk_parent.parent
    require(shutil.disk_usage(disk_parent).free>10*2**30,'Insufficient output disk space')
    if resume:
        require(read(out/'preflight.json')==meta,'Resume provenance mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'preflight.json',meta)
    os.chdir(out)
    status=dict(meta,pid=os.getpid(),started_utc=now())
    def update(**kw):
        status.update(kw,updated_utc=now())
        write_json(out/'status.json',status)
    ns={'__name__':'__main__','__file__':str(__file__),'RUN_DIR':out,
        'GRAPH_CACHE_ROOT':str(get_save_root()/'graph_cache')}
    try:
        for i in (0,1,2,4,6,7,8,9,11,12):
            update(state='preprocessing',current_cell=i)
            exec(compile(cells[i],f'{name}_{region}:cell{i}','exec'),ns)
            if i==6:
                splits={k:ns[k+'_catalogue'][:,0].astype(int).tolist() for k in ('train','val','test')}
                checker.check_splits(splits,region)
                require(not any(set(splits[a])&set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]),'Split overlap')
                write_json(out/'event_split_ids.json',splits)
            if i==9:
                write_json(out/'sampling_check.json',checker.check_test_sampling(ns,region))
        # Preserve the original second construction and all preceding RNG draws.
        model=ns['GraphNet'](activation='relu').to(ns['device'])
        ns['model']=model
        weights=state_digest(model)
        rng=rng_digest(capture_rng())
        set_encoder(model,'silu')
        require(weights==state_digest(model) and rng==rng_digest(capture_rng()),'Initial activation swap changed weights or RNG')
        reference_match=(weights==meta['expected_initial_model_digest'] and
                         rng==meta['expected_post_constructor_rng_digest'])
        if strict_reference:
            require(reference_match,'Initialization/RNG differs from original seed42 environment; strict reference check failed')
        elif not reference_match:
            warnings.warn('Initialization/RNG differs from the original environment. Seed and execution order are retained, '
                          'but bitwise reproduction is not asserted. See initialization_audit.json.', RuntimeWarning)
        init=dict(seed=42,initialization_seed=42,data_and_training_seed=42,
                  original_initial_model_digest=weights,post_constructor_rng_digest=rng,
                  weights_unchanged=True,rng_unchanged=True,replaced_activations=list(TARGETS),
                  encoder_activation='SiLU_first_11',matched_original_seed42=reference_match,
                  reference_weight_match=weights==meta['expected_initial_model_digest'],
                  reference_rng_match=rng==meta['expected_post_constructor_rng_digest'])
        if resume:
            require(read(out/'initialization_audit.json')==init,'Resume initial-state recreation mismatch')
        else:
            write_json(out/'initialization_audit.json',init)
        optimizer=torch.optim.Adam(model.parameters(),lr=POLICY['silu_lr'])
        loss_weights=torch.tensor([1.,1.,.3,.1],device=ns['device'])
        patience=meta['training_settings']['no_improve_epoch']
        require(patience==(50 if name=='gcn_max' else 70),'Unexpected patience')
        state=dict(stage='silu',epoch=0,stage_epoch=0,best_val_loss=None,best_epoch=None,
                   best_stage_epoch=None,bad_epochs=0,silu_completed=False)
        best={}
        history=[]
        if resume:
            state,best,history=restore_full(out/'checkpoints/full-state.pth',model,optimizer,meta)
            for stage,weights_saved in best.items():
                atomic_torch_save(weights_saved,out/'checkpoints'/f'{stage}-best.pth')
            if 'switch_audit' in state:
                write_json(out/'switch_audit.json',state['switch_audit'])
            logpath=out/'epochs.jsonl'
            if logpath.exists():
                stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                shutil.copy2(logpath,out/f'epochs.before_resume_{stamp}.jsonl')
            temp=out/'epochs.jsonl.tmp'
            with temp.open('w',encoding='utf-8') as f:
                for row in history:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                f.flush();os.fsync(f.fileno())
            os.replace(temp,logpath)
            write_json(out/'resumed_at.json',dict(at_utc=now(),stage=state['stage'],epoch=state['epoch']))
        batch=make_probe(ns)
        write_json(out/'initial_probe.json' if not resume else out/'resume_probe.json',probe(model,batch,ns,region))
        while True:
            if state['stage']=='silu' and state['stage_epoch']>=POLICY['silu_min_epochs'] and state['bad_epochs']>=patience:
                model.load_state_dict(best['silu'],strict=True)
                atomic_torch_save(best['silu'],out/'checkpoints/silu-best.pth')
                before_probe=probe(model,batch,ns,region)
                with preserve_rng_and_mode(model):
                    before_val=validation(model,ns['val_loader'],ns['device'],loss_weights)
                digest=state_digest(model)
                rng=rng_digest(capture_rng())
                set_encoder(model,'relu')
                require(digest==state_digest(model) and rng==rng_digest(capture_rng()),'Switch changed weights or RNG')
                after_probe=probe(model,batch,ns,region)
                with preserve_rng_and_mode(model):
                    after_val=validation(model,ns['val_loader'],ns['device'],loss_weights)
                switch=dict(from_activation='SiLU',to_activation='ReLU',replaced_activations=list(TARGETS),
                    weights_unchanged=True,rng_unchanged=True,before_weights_digest=digest,
                    after_weights_digest=state_digest(model),source_silu_best_epoch=state['best_epoch'],
                    source_silu_best_val_loss=state['best_val_loss'],switch_after_epoch=state['epoch'],
                    optimizer_reset=True,relu_lr=POLICY['relu_lr'],before_validation_loss=before_val,
                    after_validation_loss=after_val,before_probe=before_probe,after_probe=after_probe)
                write_json(out/'switch_audit.json',switch)
                state.update(stage='relu',stage_epoch=0,best_val_loss=after_val,best_epoch=state['epoch'],
                             best_stage_epoch=0,bad_epochs=0,silu_completed=True,switch_audit=switch)
                optimizer=torch.optim.Adam(model.parameters(),lr=POLICY['relu_lr'])
                best['relu']=_cpu_copy(model.state_dict())
                atomic_torch_save(best['relu'],out/'checkpoints/relu-at-switch.pth')
                atomic_torch_save(best['relu'],out/'checkpoints/relu-best.pth')
                save_full(out/'checkpoints/full-state.pth',model,optimizer,state,best,history,meta)
                update(state='training',stage='relu',epoch=state['epoch'],stage_epoch=0,
                       best_val_loss=after_val,switch_after_epoch=state['epoch'],lr=POLICY['relu_lr'])
            if state['stage']=='silu' and state['stage_epoch']>=POLICY['silu_max_epochs']:
                raise RuntimeError('SiLU did not reach the predefined stability condition within 1000 epochs; no forced switch')
            if state['stage']=='relu' and (state['stage_epoch']>=POLICY['relu_max_epochs'] or
                state['stage_epoch']>=POLICY['relu_min_epochs'] and state['bad_epochs']>=patience):
                break
            t,grads=train_epoch(model,optimizer,ns['train_loader'],ns['device'],loss_weights)
            v=validation(model,ns['val_loader'],ns['device'],loss_weights)
            state['epoch']+=1
            state['stage_epoch']+=1
            state['bad_epochs']+=1
            improved=state['best_val_loss'] is None or v<state['best_val_loss']
            if improved:
                state.update(best_val_loss=v,best_epoch=state['epoch'],best_stage_epoch=state['stage_epoch'],bad_epochs=0)
                best[state['stage']]=_cpu_copy(model.state_dict())
                atomic_torch_save(best[state['stage']],out/'checkpoints'/f"{state['stage']}-best.pth")
            p=probe(model,batch,ns,region)
            row=dict(epoch=state['epoch'],stage=state['stage'],stage_epoch=state['stage_epoch'],
                     train=t,val=v,best=state['best_val_loss'],lr=optimizer.param_groups[0]['lr'],
                     improved=improved,bad_epochs=state['bad_epochs'],first_batch_encoder_gradients=grads,probe=p)
            history.append(row)
            save_full(out/'checkpoints/full-state.pth',model,optimizer,state,best,history,meta)
            with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            update(state='training',stage=state['stage'],epoch=state['epoch'],stage_epoch=state['stage_epoch'],
                   train_loss=t,val_loss=v,best_val_loss=state['best_val_loss'],best_epoch=state['best_epoch'],
                   best_stage_epoch=state['best_stage_epoch'],bad_epochs=state['bad_epochs'],
                   lr=optimizer.param_groups[0]['lr'],probe_changed_station_vectors=p['changed_station_vectors'])
            print(json.dumps({k:row[k] for k in ('epoch','stage','stage_epoch','train','val','best','lr')}) ,flush=True)
        require(state['stage']=='relu' and state['silu_completed'],'Training ended before ReLU stage')
        # Only the ReLU validation-selected model is evaluated for final results.
        atomic_torch_save(best['relu'],out/'checkpoints/relu-best.pth')
        ns['savefile_best']=str(out/'checkpoints/relu-best.pth')
        ns['model']=model
        update(state='evaluating',stage='relu')
        exec(compile(cells[14],f'{name}_{region}:cell14','exec'),ns)
        result={}
        for split in ('train','val','test'):
            p,y=ns[split+'_P'],ns[split+'_L']
            require(np.isfinite(p).all() and np.isfinite(y).all(),'Nonfinite exported predictions')
            result[split]=metrics(y,p,region)
            if split!='train':
                ds=ns[split+'_dataset']
                ids=ds.cat[ds.event_inds,0].astype(int)
                temp=out/(split+'_predictions.tmp')
                with temp.open('wb') as f:
                    np.savez_compressed(f,event_ids=ids,predictions=p,labels=y)
                    f.flush();os.fsync(f.fileno())
                os.replace(temp,out/(split+'_predictions.npz'))
        write_json(out/'metrics.json',result)
        write_json(out/'loss_history.json',history)
        write_json(out/'final_probe.json',probe(model,batch,ns,region))
        update(state='completed',stage='relu',final_activation='ReLU',epoch=state['epoch'],
               stage_epoch=state['stage_epoch'],best_epoch=state['best_epoch'],
               best_stage_epoch=state['best_stage_epoch'],best_val_loss=state['best_val_loss'],finished_utc=now())
    except BaseException:
        update(state='failed',traceback=traceback.format_exc(),finished_utc=now())
        raise


if __name__=='__main__':
    raise SystemExit('Use python run.py --model gcn_max --region California; add --execute only to train.')
