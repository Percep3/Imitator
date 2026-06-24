# Experiments

Esta carpeta contiene los TOML de experimentos versionados. La configuracion base vive en:

- `config/model/config.toml`
- `config/training/train_config.toml`

Los runners cargan un experimento con `MSLM_EXPERIMENT_CONFIG`, por ejemplo:

```bash
MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml PYTHONPATH=. python scripts/train/train_ctc_v119.py
```

Por defecto, `src/mslm/utils/config_loader.py` carga `experiments/v115_ce_vocab/ce_vocab.toml` si no se define otra ruta.

Familias:

- `v115_ce_vocab/`: CE vocab, CE-AR y SIGReg de la linea base anterior.
- `v118_contrastive/`: pre-entrenamiento contrastivo.
- `v119_ctc/`: formulaciones CTC.
- `v120_isolated/`: clasificacion aislada v120.
- `v121_v124_isolated_staged/`: staged isolated, ablations y LOSO.
- `v126_temporal/`: pre-entrenamiento temporal por CLI args.
