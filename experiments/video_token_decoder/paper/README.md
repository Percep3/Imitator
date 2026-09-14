# Paper Imitator E1 (fuente canónica)

Documento principal: `imitator_e1.tex`. Esta es la fuente canónica del manuscrito
Springer LNCS enviado a conferencia. El contenido está dividido en `sections/*.tex`;
el archivo principal conserva el preámbulo, metadatos, orden de inclusión y
bibliografía.

Las cifras confirmatorias no se escriben a mano:
`scripts/eval/summarize_imitator_paper.py` genera `results_macros.tex` desde los
JSON de los folds 7–8. Las figuras incluidas se pueden regenerar con
`make_confirmation_figure.py` y `make_keypoint_interpretability_figure.py`.

Política de citas: `knowledge/` es el corpus local canónico. Toda afirmación relacionada
con literatura debe contrastarse primero con ese corpus. Se pueden incorporar fuentes
externas primarias verificadas cuando cubran un hueco. La cobertura archivo→BibTeX y las
fuentes externas están registradas en `KNOWLEDGE_CITATION_MAP.md`.

El relato central es la evolución de Imitator: la versión publicada imitaba embeddings
continuos; esta versión predice IDs exactos de Gemma y audita por separado si Gemma puede
realizar o corregir la secuencia. El orden temporal es evidencia técnica habilitante, no
el claim principal. La carpeta local `15/` contiene el manuscrito fuente histórico y no
forma parte del manuscrito versionado.

El alcance incluye resultados confirmatorios, protocolo LOSO, intervención causal,
interpretabilidad por keypoints, auditoría de tolerancia de Gemma y limitaciones.

La versión enviada fue compilada con Tectonic 0.16.9 y la plantilla oficial LNCS
v2.21. Desde este directorio se compila con `tectonic imitator_e1.tex`.
