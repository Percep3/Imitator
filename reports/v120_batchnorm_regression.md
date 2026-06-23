# Regresión v120: BatchNorm con inferencia B=1

El checkpoint `outputs/checkpoints/120/2/19/checkpoint.pth` no presenta una
brecha train/validation severa cuando usa estadísticas por muestra. La caída
reportada aparece al activar las estadísticas acumuladas de `BatchNorm2d`.

| Modo del mismo checkpoint | train top-1 (640) | val top-1 (640) |
|---|---:|---:|
| `model.train()` sin gradientes | 92.8% | 88.4% |
| `model.eval()` | 1.4% | 1.4% |

Esta tabla se conserva como prueba de regresión conceptual para v121: el modelo
aislado debe usar GroupNorm y sus logits deben coincidir entre `train()` y
`eval()` cuando dropout está desactivado. Los tests automatizados correspondientes
están en `tests/test_isolated_classifier.py`.
