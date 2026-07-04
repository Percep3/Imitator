# Draft del paper Imitator E1

Documento principal: `imitator_e1.tex`.

El borrador es deliberadamente neutral respecto al venue. Las cifras confirmatorias no
se escriben a mano: `results_macros.tex` contiene placeholders hasta que
`scripts/eval/summarize_imitator_paper.py` lo regenere desde los JSON de folds 7–8.

Política de citas: `knowledge/` es el corpus local canónico. Toda afirmación relacionada
con literatura debe contrastarse primero con ese corpus. Se pueden incorporar fuentes
externas primarias verificadas cuando cubran un hueco. La cobertura archivo→BibTeX y las
fuentes externas están registradas en `KNOWLEDGE_CITATION_MAP.md`.

Partes ya defendibles: task, alcance, datos sintéticos, arquitectura, protocolo LOSO,
métricas, hipótesis, intervención causal, resultados de desarrollo y limitaciones.

Pendiente antes de submission:

- resultados confirmatorios y sus CIs;
- curvas secundarias si entran en el presupuesto;
- documentación ética/licencia/demografía de la fuente de datos;
- autores, venue y plantilla final;
- revisión bibliográfica más amplia.

El entorno actual no incluye un compilador TeX; el archivo se valida aquí mediante
estructura y tests del generador de macros, y debe compilarse al adoptar la plantilla del
venue.
