# Auditoría lingüística — dataset1 vs dataset2 (v125)

## Hallazgo

- `dataset2` corresponde a LSA (Lengua de Señas Argentina). La fuente queda
  identificada en `data/raw/dataset2/meta.csv` mediante programas como
  `noticias-en-lengua-de-senas-argentina-resumen-semanal-*`; el archivo también
  conserva programa, playlist, segmentos y signer inferido.
- La lengua de origen de `dataset1` no está documentada en el repositorio.
  `data/raw/dataset1/meta.csv` sólo contiene `id,label`; `label.csv` contiene
  glosas en español; los nombres `<clase>_<signer>_<repetición>` permiten
  recuperar identidad experimental, pero no la variante lingüística ni la
  fuente original.

## Decisión

No se puede afirmar que ambos datasets pertenezcan a la misma lengua de señas.
Por precaución, v126+ sólo puede cargar del checkpoint v121 los pesos de
extracción corporal de bajo nivel:

- `stgcn_layers.*`
- `linear_hidden.*`

La ruta verificada del checkpoint de referencia es
`/shared/Code/Sign-AI/outputs/checkpoints/121/23/best_top1/checkpoint.pth`.

No se debe cargar `classifier.*`, transferir la cabeza de 64 clases ni asumir
equivalencia semántica entre las glosas aisladas de `dataset1` y las
transcripciones continuas de `dataset2`.

## Signers

- `dataset1`: diez signers recuperables desde el segundo campo del nombre del
  clip; esta identidad sí puede usarse para evaluación leave-one-signer-out.
- `dataset2`: `meta.csv` aporta un signer inferido por segmento y confianza,
  pero no una identidad estable y fiable entre programas. Por ahora no se usa
  como segunda restricción del split agrupado.

## Pendiente

Si se identifica la publicación o dataset original de `dataset1`, actualizar
esta auditoría antes de ampliar la transferencia más allá de rasgos visuales
de bajo nivel.
