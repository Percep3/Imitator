# Comandos de reproducción: auditoría dataset2

- Fecha de la auditoría: 2026-10-19. Repo en HEAD `a7db233a86eff014459bedecb470e1a7c4141add`.
- Todos los comandos se ejecutan desde `/shared/Code/Sign-AI/Sign-chris`.
- Todos son de solo lectura sobre los datos: el HDF5 se abre con `"r"` y los videos solo se sondean con ffprobe. Solo escriben bajo `artifacts/dataset2_audit/`.

## Entorno

```bash
cd /shared/Code/Sign-AI/Sign-chris
PY=/shared/Code/nakato/projects/VQ_RL/.venv/bin/python   # py3.13, h5py 3.16, numpy 2.3.1, torch 2.13 (CPU), tokenizers 0.22.2
# Sign-env (~/miniconda3/envs/Sign-env) está vacío (solo conda-meta/). NO instalar nada en él.
# ffprobe 6.1.1 en /usr/bin/ffprobe (no hay cv2 en ninguna env).
export HDF5_USE_FILE_LOCKING=FALSE
```

Entradas (sha256 esperados, iguales a `/shared/Code/Sign-AI/data/processed/v125_artifacts.json`):

```
24c9aa401b44b14283ea71851b12ccb44b3c224ec7d7b85119b534389c3b1d38  /shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5
04c3a7c1836483bc7fcd749b32386339b5e407be34d9483bb4e1ec3276598bad  /shared/Code/Sign-AI/data/processed/dataset2_manifest_v125.json
6669635ad97950c4f48c90816f8bf99e5902bd107412cc049670bec6f388dbf9  /shared/Code/Sign-AI/data/processed/dataset2_split_v125.json
2043797146e6d6ff94d107f25715fe602d73429adf74ab79471818be2763a6d7  /shared/Code/Sign-AI/data/processed/gemma3n_embed_table.pt
b6c35ee648c07754b44cd9e371c75d4caa05c4504910b7ad29b1847ee9d8ba5d  ~/.cache/huggingface/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30/tokenizer.json
```

Para comprobarlos, pega las líneas anteriores en `inputs.sha256`, cambia `~` por `$HOME` y ejecuta `sha256sum -c inputs.sha256`.

## 1. Procedencia e inventario (`_work/01_*`)

```bash
python3 artifacts/dataset2_audit/scripts/inventory.py --full-big > artifacts/dataset2_audit/_work/01_inventory.tsv
$PY artifacts/dataset2_audit/scripts/h5_provenance.py > artifacts/dataset2_audit/_work/01_h5_provenance.json
# meta.csv histórico (solo 10 filas visibles):
git show fef07a3:Sign-Dataset.ipynb | python3 -c 'import json,sys; [print("".join(o.get("text",""))) for c in json.load(sys.stdin)["cells"] for o in c.get("outputs",[])]'
# procedencia del "3088":
git log --all -S3088 --oneline; grep -rn check_alignment src scripts
ls /shared/Code/Sign-AI/data/raw/dataset2/videos | sed -E 's/_[0-9]+\.mp4$//' | sort | uniq -c | sort -rn | wc -l   # 64 (+1 por la línea ._*)
```

## 2. Integridad visual (`_work/visual_*`, unos 65 s con 10 hilos de ffprobe)

```bash
$PY artifacts/dataset2_audit/scripts/visual_integrity.py
```

## 3. Texto, splits y fuga (`_work/text_split_*`, unos 75 s en CPU)

```bash
$PY artifacts/dataset2_audit/scripts/text_split_audit.py
```

## 4. Experimentos históricos (`_work/04_experiments_split_stats.json`)

```bash
$PY artifacts/dataset2_audit/scripts/experiment_split_leakage.py
python3 artifacts/dataset2_audit/scripts/tb_scalars.py /shared/Code/Sign-AI/outputs/reports/117 infonce   # Loss/train_infonce max=0
# v116 ≡ v117: ver _work/04_experiments_impact.md §7.1 (esperado: "354 350 {'Gen/val_chrf'}")
```

## 5. Muestra para revisión humana (determinista; `sample_manifest.csv`, `human_review_template.csv`, `_private_previews/pilot/`)

```bash
$PY artifacts/dataset2_audit/scripts/build_review_sample.py
```

## 6. Informe de integridad por clip y estadísticas fusionadas

Este paso depende de los pasos 2, 3 y 5.

```bash
$PY artifacts/dataset2_audit/scripts/merge_integrity_report.py     # -> integrity_report.csv (5.600 filas)
$PY artifacts/dataset2_audit/scripts/build_dataset_statistics.py   # -> dataset_statistics.json
```

Salida esperada de `merge_integrity_report.py`:

```
{'kept_warn': 1548, 'kept_no_warn': 3590, 'excluded_v125:truncated': 413, 'excluded_v125:frames_per_token': 16, 'excluded_v125:min_tokens': 29, 'excluded_v125:min_frames': 4}
```

## 7. Comprobación de la contradicción del tope 250/1400

```bash
$PY - <<'X'
import csv
v=list(csv.DictReader(open('artifacts/dataset2_audit/_work/visual_integrity_per_clip.csv')))
for cap in (250,1400):
    c=[r for r in v if int(r['frame_count_kp'])==cap]
    print(cap,len(c),sorted({r['video_nb_frames'] for r in c}))
print('videos>1400 con T<1400:',sum(int(r['video_nb_frames'])>1400 and int(r['frame_count_kp'])<1400 for r in v))
X
# esperado: 250 -> 12 (nb_frames 250 o 270); 1400 -> 12 (todos con nb_frames > 1400); 0
```

## 8. Hashes esperados de las salidas

```
8b72d8a4caaf60693313451af4836913156bbb1ecf4847771d3dd2136a7a5e49  artifacts/dataset2_audit/integrity_report.csv
18626cbbe774bfc9540b3be6bf769539604e2249e243a8de8e954326f43470fb  artifacts/dataset2_audit/dataset_statistics.json
5ba98a5243b29a301fb0ce112acdbc14daac3dbaa68955d072b362816931de13  artifacts/dataset2_audit/sample_manifest.csv
89a5d9fa03f1aa2fc96c0bd3fcc8c7ae6bf7e0223b11609a2cb12afa69ab21ef  artifacts/dataset2_audit/human_review_template.csv
38040b42c668bfd6b10a99b84dc374cce5ca22b5b943b436c2a40fe93a5887d0  artifacts/dataset2_audit/_work/visual_integrity_per_clip.csv
c6f16fce50b9c6bca15eb6736e61d360d9b4dfd4ea50811a2b2decd52f07b248  artifacts/dataset2_audit/_work/text_split_per_clip.csv
036ae3522f7533dbd4ab02ea40bd83f2c723e4653b206c3fee0acb1260077ab7  artifacts/dataset2_audit/_work/visual_stats.json
f6ab42bf651d184c7088283887ed24320857a0ce58c08ffead3b28ef2c47782a  artifacts/dataset2_audit/_work/text_split_stats.json
1349e854cdc30525f8ff87dca17bd4dfa22a34a9fc4416fc970d3765458f965c  artifacts/dataset2_audit/_work/01_h5_provenance.json
5e80ce6f8d68e3a9a4a874cf87ada94ccf6c48d9f75f0b8e565110dced0c0cdd  artifacts/dataset2_audit/_work/04_experiments_split_stats.json
```

Para verificar, guarda el bloque anterior en `outputs.sha256` y ejecuta `sha256sum -c outputs.sha256`.

Advertencias:
- Solo `sample_manifest.csv`, `human_review_template.csv` y `dataset_statistics.json` se han comprobado byte a byte en dos ejecuciones. Los CSV y JSON de `_work/` no se re-ejecutaron para comprobar su determinismo byte a byte; sus valores sí se verificaron.
- `dataset_statistics.json` embebe el commit (`git rev-parse HEAD`) y los sha256 de los artefactos de trabajo. Su hash cambia si cambia HEAD o cualquiera de esas entradas.
- `inventory.py` registra mtimes. `01_inventory.tsv` cambia si se tocan los archivos de datos.
- `experiment_split_leakage.py` usa `torch.randperm`, con torch 2.13 aquí frente a 2.8 en el entrenamiento histórico. Los tamaños coinciden, pero la identidad de la permutación es INFERRED.

_Generado con asistencia de IA (Claude)._
