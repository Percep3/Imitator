# Multimodal Siign Language Model
This project is a small model, designed to extend Llama's skills, equipping him with a knowledge of sign language, through a small model that maps video signs to text embeddings.

## Project layout

- `src/mslm/`: libreria principal.
- `scripts/train/`: entrypoints de entrenamiento y orquestacion.
- `scripts/data/`: builders, backfills y manifiestos de datos.
- `scripts/diagnostics/`: diagnosticos de modelos y datos.
- `scripts/audits/`: auditorias v125/Gemma/freeze/smoke.
- `experiments/`: TOML versionados por familia experimental.
- `config/`: configuracion base compartida.

Comandos canonicos:

```bash
PYTHONPATH=. python scripts/train/train_imitator_dataset1_tokens.py
PYTHONPATH=. python scripts/train/train_temporal_v126.py
PYTHONPATH=. python scripts/train/train_isolated_staged.py --config experiments/v121_v124_isolated_staged/cls_v121.toml
MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml PYTHONPATH=. python scripts/train/train_ctc_v119.py
```

Objetivo Imitator actual: el modelo oficial no es la clasificacion 64-way de
glosas aisladas. El prototipo real predice **token IDs de Gemma** desde
video/keypoints de dataset1 y exporta `predictions.jsonl` con token IDs, texto
decodificado y prompt v125 para que Gemma corrija/formatee la salida final.
