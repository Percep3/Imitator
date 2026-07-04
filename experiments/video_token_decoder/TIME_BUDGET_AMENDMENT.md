# Enmienda operativa por presupuesto — 2026-07-04

Esta enmienda se registra después del pre-registro (`67641eb`) y antes de computar u
observar resultados outer-test E1 de folds 7–8. No cambia la configuración E1, las 30
épocas, la seed, las muestras, las métricas, los umbrales ni las reglas H1–H3.

El presupuesto operativo se redujo a aproximadamente 24 horas, con una tolerancia de
2–3 horas. Para priorizar la evidencia confirmatoria se aplazan:

- el retrain/validación de robustez en fold 4;
- jitter, frame dropout y remuestreo temporal secundarios;
- probes lineales post-hoc;
- E3 exploratorio en folds 5–6.

Se mantienen ambos folds confirmatorios. La primera entrega ejecutará únicamente:

1. preparación hasheada de los inits CIF de folds 7–8;
2. comparadores CIF pareados;
3. E1 exacto, 30 epochs, seed 23, en ambos folds;
4. outer-test limpio y permutación gold-boundary en ambos folds;
5. H1–H3 y tabla maestra.

Clean + permutación contienen toda la evidencia necesaria para H1–H3. La batería
secundaria y los análisis exploratorios se podrán ejecutar posteriormente sobre los
mismos checkpoints cerrados, sin reentrenar E1. Esta reducción afecta completitud
descriptiva, no la decisión confirmatoria pre-registrada.
