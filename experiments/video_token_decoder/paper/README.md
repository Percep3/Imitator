# Draft del paper Imitator E1

Documento principal: `imitator_e1.tex`. El contenido está dividido en
`sections/*.tex`; el archivo principal conserva únicamente preámbulo, metadatos,
orden de inclusión y bibliografía para facilitar el cambio de plantilla.

El borrador es deliberadamente neutral respecto al venue. Las cifras confirmatorias no
se escriben a mano: `results_macros.tex` contiene placeholders hasta que
`scripts/eval/summarize_imitator_paper.py` lo regenere desde los JSON de folds 7–8.

Política de citas: `knowledge/` es el corpus local canónico. Toda afirmación relacionada
con literatura debe contrastarse primero con ese corpus. Se pueden incorporar fuentes
externas primarias verificadas cuando cubran un hueco. La cobertura archivo→BibTeX y las
fuentes externas están registradas en `KNOWLEDGE_CITATION_MAP.md`.

El relato central es la evolución de Imitator: la versión publicada imitaba embeddings
continuos; esta versión predice IDs exactos de Gemma y audita por separado si Gemma puede
realizar o corregir la secuencia. El orden temporal es evidencia técnica habilitante, no
el claim principal. La carpeta local `15/` contiene el manuscrito fuente histórico y no
forma parte del borrador versionado.

Partes ya defendibles: relación con Imitator v1, task, alcance, datos sintéticos,
arquitectura, protocolo LOSO, métricas, hipótesis, intervención causal, auditoría de
tolerancia de Gemma, resultados de desarrollo y limitaciones.

Pendiente antes de submission:

- resultados confirmatorios y sus CIs;
- curvas secundarias si entran en el presupuesto;
- detalles de consentimiento/demografía que no aparecen en la documentación pública
  de LSA64;
- venue y plantilla final;
- revisión bibliográfica más amplia.

El entorno actual no incluye un compilador TeX; el archivo se valida aquí mediante
estructura y tests del generador de macros, y debe compilarse al adoptar la plantilla del
venue.
