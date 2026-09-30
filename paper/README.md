# Imitator papers

Esta carpeta separa el contenido científico de las plantillas editoriales.
El manuscrito actual es un **full paper** de hasta 11 páginas en Springer LNCS; la
versión short todavía no ha sido redactada.

## Organización

- `content/full/`: texto universal del full paper, dividido por secciones.
- `content/short/`: futura reducción editorial, independiente del full paper.
- `shared/`: metadatos, bibliografía, macros de resultados y mapa de citas.
- `formats/`: wrappers y plantillas propios de cada formato editorial.
- `figures/`: figuras finales usadas por los manuscritos.
- `data/frozen-paper-results/`: entradas congeladas para tablas y figuras.
- `tools/figures/`: scripts de generación de figuras.
- `submissions/`: PDFs efectivamente enviados, conservados como snapshots.
- `build/`: salidas locales reproducibles; Git las ignora.
- `archive/`: manuscritos históricos que no corresponden al trabajo actual.

## Compilación Springer full

Se requiere Tectonic 0.16.9. Desde la raíz del repositorio:

```bash
TECTONIC_BIN=/ruta/a/tectonic paper/build.sh springer-lncs full
```

El script fija por defecto `SOURCE_DATE_EPOCH=1783312141`, usado para reproducir
el PDF enviado. La salida queda en
`paper/build/springer-lncs/full/imitator_e1.pdf`. El enlace
`formats/springer-lncs/imitator_e1.tex` conserva el nombre de trabajo usado en
la entrega; sin él solo cambia el ID interno del PDF y, por tanto, su SHA.

SHA-256 del snapshot previamente enviado (no se sobrescribe al regenerar):

```text
bb6edb38ef200da7479e25aafdd5097fbb5805d0730690bcf7ea4d69fdf1a23f
```
