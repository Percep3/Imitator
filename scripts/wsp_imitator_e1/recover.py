"""Recovery TRAINING only, separate from the gradient-free WSP experiment.

Runs the repository's original v121 -> CIF -> E1 training chain in isolated
output directories. No outer evaluation. Checkpoint bytes will be NEW artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "outputs/wsp_imitator_e1/recovery"
TOKENIZER = Path("/home/nakato/.cache/huggingface/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def record(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,default=str)+'\n')
    temporary.replace(path)


def initialize_direct_e1(model, checkpoint_path, *, outer_signer, expected_sha256=None):
    """Training initialization: trained v121 spatial encoder, fresh AR decoder."""
    from scripts.wsp_imitator_e1.harness import torch
    from src.mslm.models.video_token_decoder import validate_checkpoint_lineage
    actual = sha(checkpoint_path)
    if actual != expected_sha256:
        raise ValueError('v121 checkpoint hash mismatch')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    lineage = validate_checkpoint_lineage(state, outer_signer)
    source = state['model_state']
    loaded = []
    for name in ('stgcn_layers', 'linear_hidden'):
        module = getattr(model.frame_encoder, name)
        prefix = name + '.'
        subset = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
        module.load_state_dict(subset, strict=True)
        loaded.extend(prefix + k for k in subset)
    # Standard small embedding initialization for a newly trained Transformer.
    # This is performed before training, never as a WSP edit to trained weights.
    with torch.no_grad():
        torch.nn.init.normal_(model.token_embedding.weight, mean=0., std=.02)
        model.token_embedding.weight[model.pad_id].zero_()
    return dict(checkpoint_sha256=actual, lineage=lineage,
                initialization_kind='v121_visual_only_fresh_autoregressive_decoder',
                initialized_from_v121=loaded, token_embedding_initial_std=.02)


def stage(fold, name, threads):
    # Set config environment before importing the repo ConfigLoader singleton.
    config = RECOVERY / f"fold{fold}" / 'cls_v121.toml'
    if name == 'v121':
        os.environ['MSLM_EXPERIMENT_CONFIG'] = str(config)
    from scripts.wsp_imitator_e1.harness import repo, permitted_records, torch
    from src.mslm.dataloader import isolated_keypoint_dataset as isolated
    from scripts.loso.run_clean_loso_orchestrator import STAGE_SPECS, expected_run_name
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    spec = repo.fold_spec(repo.load_manifest(repo.DEFAULT_MANIFEST), fold)
    if fold not in (4,5,6):
        raise ValueError('Recovery only permits folds 4-6')
    # Filter signer membership BEFORE labels, including in the original trainers.
    # Pool order matches original numeric-clip order after train/val filtering.
    def allowed_records(h5, dataset_name='dataset1'):
        if dataset_name != 'dataset1':
            raise ValueError('dataset1 only')
        rows = permitted_records(h5,spec,'discovery') + permitted_records(h5,spec,'confirmation')
        import h5py
        with h5py.File(h5, 'r') as handle:
            repetitions = handle[dataset_name]['repetition']
            for row in rows:
                row['repetition'] = int(repetitions[row['clip_id']][0])
        return sorted(rows,key=lambda row:int(row['clip_id']))
    isolated.list_clip_records = allowed_records
    repo.list_clip_records = allowed_records
    root = RECOVERY / f'fold{fold}'
    v121 = root / f'isolated/checkpoints/121/{9000+fold}/best_top1/checkpoint.pth'
    common = ['--heldout-signer',str(spec['inner_val_signer']), '--exclude-signer',str(fold),'--seed','23']
    if name == 'e1_direct':
        if spec['train_signers'] != torch.load(v121, map_location='cpu', weights_only=True)['lineage']['train_signers']:
            raise ValueError('v121 training signer list differs from manifest')
        tag = 'e1_direct_v121'
        target = root / 'ar' / f'fold{fold}_seed23' / tag
        if target.exists():
            raise RuntimeError(f'direct E1 output already exists; inspect before restart: {target}')
        repo.initialize_from_cif_checkpoint = initialize_direct_e1
        args = repo.parse_args(['train','--fold',str(fold),'--seed','23','--epochs','30',
            '--encoder-pe','--restricted-vocab','--label-smoothing','0','--select','edit',
            '--run-tag',tag,'--tokenizer',str(TOKENIZER),
            '--source-checkpoint',str(v121),'--output-root',str(root/'ar')])
        repo.command_train(args)
        return
    if name == 'v121':
        sys.argv = ['scripts/train/train_isolated_staged.py','--config',str(config),
                    '--h5-filename','dataset1_isolated_v122.hdf5','--run-id',str(9000+fold),*common]
        runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        return
    if name == 'e1':
        source = root / 'temporal' / f'diag_fold{fold}_promotion/checkpoint_best.pt'
        args = repo.parse_args(['train','--fold',str(fold),'--seed','23','--epochs','30',
            '--encoder-pe','--restricted-vocab','--label-smoothing','0','--select','edit',
            '--run-tag','e1_retrained_original_chain','--tokenizer',str(TOKENIZER),
            '--source-checkpoint',str(source),'--output-root',str(root/'ar')])
        repo.command_train(args)
        return
    from scripts.train import train_temporal_v126 as temporal
    spec_index = next(i for i,s in enumerate(STAGE_SPECS) if s['name']==name)
    spec_stage = STAGE_SPECS[spec_index]
    argv = [*common,'--epochs',str(spec_stage['epochs']),'--output-root',str(root/'temporal'),
            '--run-name',f'fold{fold}_{name}', '--checkpoint-v121',str(v121),
            '--tokenizer',str(TOKENIZER),*spec_stage['extra']]
    if spec_index:
        previous = STAGE_SPECS[spec_index-1]
        parent_dir = expected_run_name(f"fold{fold}_{previous['name']}",previous['extra'])
        argv += ['--resume',str(root/'temporal'/parent_dir/'checkpoint_best.pt'),'--resume-weights-only']
    # Existing A1 metadata omits the v121 initializer; retain it explicitly in NEW
    # recovery checkpoints so the eventual WSP ancestry audit can reach the root.
    original_lineage = temporal.build_lineage
    def lineage(args, train, val):
        data = original_lineage(args,train,val)
        if not data['parent_checkpoint']:
            data['parent_checkpoint'] = str(v121)
            data['parent_checkpoint_sha256'] = sha(v121)
        data['recovery_training'] = True
        return data
    temporal.build_lineage = lineage
    sys.argv=['scripts/train/train_temporal_v126.py',*argv]
    temporal.main()


def chain(folds,threads):
    from scripts.loso.run_clean_loso_orchestrator import STAGE_SPECS, expected_run_name
    RECOVERY.mkdir(parents=True,exist_ok=True)
    for fold in folds:
        if fold not in (4,5,6):
            raise ValueError('Only folds 4-6')
        root=RECOVERY/f'fold{fold}'
        root.mkdir(exist_ok=True)
        cfg=(ROOT/'experiments/v121_v124_isolated_staged/cls_v121.toml').read_text()
        cfg=cfg.replace('output_root = "../outputs"',f'output_root = "{root}/isolated"')
        config=root/'cls_v121.toml'
        if config.exists() and config.read_text()!=cfg:
            raise ValueError('recovery config changed')
        config.write_text(cfg)
        stages=['v121',*[s['name'] for s in STAGE_SPECS],'e1']
        for name in stages:
            if name=='v121':
                checkpoint=root/f'isolated/checkpoints/121/{9000+fold}/best_top1/checkpoint.pth'
            elif name=='e1':
                checkpoint=root/f'ar/fold{fold}_seed23/e1_retrained_original_chain/checkpoint_closed.pt'
            else:
                s=next(s for s in STAGE_SPECS if s['name']==name)
                checkpoint=root/'temporal'/expected_run_name(f'fold{fold}_{name}',s['extra'])/'checkpoint_best.pt'
            done=root/f'{name}_done.json'
            if done.exists():
                previous=json.loads(done.read_text())
                if checkpoint.is_file() and sha(checkpoint)==previous['checkpoint_sha256']:
                    continue
                raise RuntimeError(f'completed recovery artifact changed or missing: {checkpoint}')
            command=[sys.executable,'-u','-m','scripts.wsp_imitator_e1.recover','stage',
                     '--folds',str(fold),'--stage',name,'--threads',str(threads)]
            log=root/f'{name}.log'
            # Never overwrite a previous interrupted stage without an explicit audit.
            if log.exists():
                raise RuntimeError(f'incomplete stage log exists; inspect before restart: {log}')
            with log.open('x') as output:
                process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':str(threads),
                         'MKL_NUM_THREADS':str(threads),'PYTHONHASHSEED':'23'})
                record(RECOVERY/'job.json',dict(status='running',fold=fold,stage=name,pid=process.pid,
                    coordinator_pid=os.getpid(),command=command,log=str(log),started=time.time(),
                    original_historical_checkpoint=False,training_outside_wsp_budget=True))
                code=process.wait()
            if code:
                record(RECOVERY/'job.json',dict(status='failed',fold=fold,stage=name,exit_code=code,log=str(log)))
                raise RuntimeError(f'recovery stage {name} failed: {code}')
            result=dict(fold=fold,stage=name,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                        original_historical_checkpoint=False,completed=time.time())
            record(done,result)
            if name!='e1':
                record(checkpoint.parent/'stage_done.json',result)
            print(json.dumps(result),flush=True)
    record(RECOVERY/'job.json',dict(status='complete',folds=folds,completed=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['chain','stage'])
    p.add_argument('--folds',type=int,nargs='+',default=[4,5,6])
    p.add_argument('--stage')
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    if args.command=='stage':
        if len(args.folds)!=1:
            p.error('one fold per stage')
        stage(args.folds[0],args.stage,args.threads)
    else:
        chain(args.folds,args.threads)

if __name__=='__main__':
    main()
