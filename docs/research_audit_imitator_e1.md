# Auditoría histórica de investigación de Imitator-E1

## 1. Executive summary

**No se encontró evidencia verificable de éxito en traducción directa de video continuo natural a oraciones.** Sí hubo intentos de generación con Gemma congelado sobre `dataset2` (v116–v117), preentrenamiento contrastivo (v118), CTC sobre transcripciones españolas (v119), reconocimiento aislado (v120–v122), CIF/token imitation (v126) y generación autorregresiva de IDs Gemma sobre concatenaciones artificiales (VideoTokenDecoder/E1). Son tareas diferentes [E03–E13].

**VERIFIED:** E1 obtiene strict exact 12,50% y 14,06% en los folds confirmatorios 7 y 8: 896 secuencias por fold, seed 23. Supera a CIF affine con 99/3 y 108/6 wins/losses. Permutar segmentos reduce edit similarity de 0,530/0,587 a 0,188/0,209. Existe dependencia del orden visual en esta tarea sintética; no demuestra sintaxis, coarticulación ni traducción abierta [E11–E13].

**Hallazgo nuevo:** v117 registra `Loss/train_infonce=0.0` en las 25 épocas. El código histórico calcula InfoNCE dentro de cada sub-batch; con B=1 su matriz 1×1 tiene CE y gradiente cero. La afirmación histórica de que InfoNCE retrasó el colapso no tiene respaldo causal en esa corrida. v118 sí reúne las representaciones del batch antes del contraste y constituye una prueba activa diferente [E05–E07].

**Otras correcciones:** la CE 0,568 de v115.1 es imposible con label smoothing 0,1 y 262.400 clases. Un commit posterior identifica doble división por sub-batches. El LOSO CIF de ~90,5% quedó anulado por exposición del signer de test en etapas anteriores. La cadena limpia rindió ~0,37% en su antigua métrica `exact`, que además realinea logits al target [E04, E10].

**Dataset2 no está certificado como supervisión confiable de SLT.** Hay videos de noticias LSA con texto español y un filtro que conserva 5.138/5.600 clips. Este filtro no certifica correspondencia semántica ni desfase entre interpretación y subtítulos. Los metadatos crudos ya no están disponibles; el HDF5 y manifiestos sí existen y sus hashes coinciden con v125. La unicidad de frases no demuestra inviabilidad de seq2seq ni explica por sí sola el fracaso CTC [E08, E14–E16].

No se encontraron implementaciones hiperbólicas en los 54 tips de ramas inspeccionados de Imitator. Geo-Sign implementa una rama Poincaré junto a mT5; su eficacia no se transfiere automáticamente a este proyecto [E18].

La primera inversión científica debe ser certificar una muestra continua de `dataset2`. Dentro de E1 falta aislar PE de vocabulario, smoothing y presupuesto: el paquete cambió todo simultáneamente. No hay pesos experimentales en `../outputs`; reutilizarlos exige restaurar checkpoints, aunque permanecen predicciones, configuraciones y linaje [E12, E17].

## 2. Repository provenance

- Fecha de auditoría: **2026-10-19**. Inspección estática y lectura de JSON, JSONL, CSV, notebooks y TensorBoard. Sin entrenamiento, inferencia GPU ni instalación de dependencias.
- Repositorio: `/shared/Code/Sign-AI/Sign-chris`.
- Branch: `v125-temporal-prompt-alignment`.
- HEAD: `a7db233a86eff014459bedecb470e1a7c4141add`, fechado `2026-09-30T19:11:52-05:00`.
- Git: 687 commits alcanzables; 54 referencias de ramas locales/remotas; ningún tag devuelto por `git tag`. No se hizo fetch: los tips remotos son los disponibles localmente.
- Ramas locales: `MHA2`, `STGCN`, `ce-vocab`, `develop`, `feat/Imitator`, `finalBoss`, `gcn`, `gemma`, `main`, `refactor`, `rope`, `sigreg`, `transformer`, `v119-ctc-sequence`, `v125-temporal-prompt-alignment`, `verde`.
- Estado previo: eliminación local de `paper/revisions/Sign-chris-1tl/README.md` y archivo no seguido `academic_writing_avoid_ai_style_signals.md`. No se tocaron. Se conservaron los cuatro stashes existentes.
- Única escritura: este informe. Se leyó `bd prime`; no se modificaron issues, commits ni remotos porque la instrucción específica solo autoriza escribir el informe. No se compiló el paper: produciría escrituras adicionales. No se cambió ningún manuscrito; **el límite de 11 páginas no fue revalidado**.

**Convenciones.** `VERIFIED`: código y artefactos accesibles respaldan el hecho concreto; no significa repetir entrenamiento. Las comprobaciones de Git, existencia y hashes se identifican como verificaciones de lectura. `DOCUMENTED`: reporte/configuración sin ejecución suficiente verificable. `INFERRED`: deducción o hipótesis causal. `UNKNOWN`: evidencia insuficiente. Una cifra puede estar VERIFIED como registro y ser científicamente inválida. Las referencias `[E##]` se resuelven en §13 mediante líneas, commits o claves de artefactos. Las rutas `../data` y `../outputs` parten del repositorio.

## 3. Historical timeline

Las fechas de commits indican integración, no necesariamente inicio del run. Commits duplicados con igual mensaje no constituyen réplicas experimentales.

| Fecha/commit | Versión | Objetivo | Arquitectura | Dataset | Cambios | Resultado | Decisión |
|---|---|---|---|---|---|---|---|
| 2025-03-23 `8dc200b` | prototipo | aprender interfaz Llama | notebook Transformer/MSE | UNKNOWN | tabla Llama-3.2-3 B y vecinos coseno | código incompleto: `fordward`, reader `...`; ejecución no validada | explorar embeddings [E01] |
| 2025-05-03 `2fb8a78`; 05-23 `ce29be5`; 07-18 `f13a090` | GCN/Transformer/RoPE | representación visual | encoders sucesivos | HDF5 históricos | GCN, regreso Transformer, RoPE | relación run→commit UNKNOWN | no atribuir métricas actuales [E01, E02] |
| 2025-08-03 `2fa6d43`; 10-15 `6350654` | STGCN/queries | embeddings por posición | STGCN + Transformer RoPE + atención | histórico | STGCN y correcciones de máscaras/RoPE | implementado; resultado ligado al commit UNKNOWN | base heredada [E02] |
| 2025-10-06 `fccb8e1`,`ecbbd29`; 10-09 `c200305` | contraste/soft-token | alinear video-texto | InfoNCE y nuevas cabezas | histórico | máscara de negativos de igual etiqueta | implementación; evaluación libre trazable UNKNOWN | InfoNCE antecede a v117 [E02] |
| 2026-06-09 `d1c95a4` | SignAlign | seq2seq de embeddings | GCN/BiLSTM + Transformer decoder | UNKNOWN | teacher forcing y realimentación vectorial | prototipo sin run demostrado | separado de E1 [E03] |
| 06-09/10 `9b085d5`,`ef12577` | v113/v114 | evitar colapso | Imitator MSE+cos; +SIGReg | ds2, 600 reportados | regularización gaussiana | effrank final 1,78/4,03; SIGReg activo en TB | pasar a CE [E04, E06] |
| 06-10 `2a70917` | v115 | clasificación de IDs | queries→tabla Gemma | ds2 | dot-product CE | runaway de norma; CE con escalado defectuoso | normalizar logits [E04] |
| 06-10; fix 06-15 `7b3eae0` | v115.1 | controlar confianza | cosine CE/temp | ds2 | smoothing y best checkpoint | CE 0,568 corrupta; top1 ~10,87% documentado | reparar sub-batches [E04] |
| 06-15 `1470d4a`,`be8e9f2` | v116/116.1 | generar frases | PrefixImitator→Gemma frozen | ds2 481/119 | primer token, padding, escala; SIGReg | v116 chrF 20,671; colapso; v116.1 pico observado 19,150 | medir grounding [E05, E06] |
| 06-18 `eb97677` | v117 | contraste auxiliar | mismo soft prefix | ds2 481/119 | InfoNCE, selección chrF, shuffle | chrF 20,601; **NCE=0 en todas las épocas** | fracaso visual; NCE activo no probado [E05, E06] |
| 06-18 `07ec4cf`; 06-19 `daeb58b` | v118–f | retrieval sin LM | dos torres/proyecciones | ds2 600→5600 | InfoNCE/VICReg, aug, texto contextual | R@1 1,68–2,94%/119; f 0,268%/1119 | giro a secuencias [E07] |
| 06-20 `a48b9fe` | v119 inicial | CTC | STGCN/RoPE/frame classifier | ds2 subset/full | palabras, sin LM | WER ~99,5–100% documentado | sustituir backend [E08] |
| 06-21 `7962692`,`403cfe4` | v119 reemplazo | supervisión dual | GCN→TCN/TLP→BiLSTM | ds2 5600 | CTC dual, filtros/aug, ambas cabezas | WER ~99–105%; menor variación tras BiLSTM | pivote a aislado [E08] |
| 06-22 `f088d51`; 06-23 `35a49d5` | v120–124 | validar encoder | STGCN/GN; variantes | ds1 3200 | BN→GN, trim y streams | v121 94,53±0,80% val; v122 peor | detener full runs v123/v124 [E09] |
| 06-23 `d8923cf`,`41539aa`,`482f4e8`,`5977e40` | v125 | certificar datos e interfaz | transform/oracle Gemma | ds2 5600→5138 | split agrupado, PLE y prompts | PCA falla; oracle chrF 92,865 con texto GT | full_standardized, no éxito visual [E14–E16] |
| 06-24 `3244979`,`15b7acc` | v126 inicial | segmentación | STGCN/TCN/Transformer+CIF | ds1 concatenado | teacher→learned alpha | colapso a cero fires al mezclar pronto | warmup teacher-only [E10] |
| 06-25–28 `f1ae9a1`,`7ff6015` | A1/A2/A3/A4 | tokens y longitud | CIF y cabezas auxiliares | ds1; muchos runs de **un clip** | Optuna, factorial24 y scheduled alpha | head contextual mejora; mezcla sin ganancia deployable | promover pre-mezcla [E10, E17] |
| 06-28 `a8ed270`; 06-30 `4badce7`; 07-03 `13c9c43` | LOSO | generalización | cadena completa | ds1 | excluir outer desde pretraining | 90,53% anulado; limpio exact medio 0,0037 | diagnosticar longitud [E10] |
| 07-03 `2e48f62`; 07-04 `aa548d4` | rounded/affine CIF | corregir conteo | mismo modelo | sintético LOSO | round(sum α), bias y affine | comparadores exact 1,8–4,9% según fold | baseline AR; ya probado [E10, E12] |
| 07-04 `abbed25` | AR-base/E1/E2/E3 | IDs ordenados | encoder CIF + decoder propio | concat ds1 2–8 | PE/vocab121/LS0/30 ep; CTC/unfreeze | E1 dev 16,9–19,8%; E2 empate; E3 exploratorio | elegir E1 por inner-val [E11, E12] |
| 07-04 `67641eb`,`3c0f703`; resultados `023386f` | confirmación | congelar E1 | misma variante | folds 7–8 | prerregistro/enmienda | 12,5/14,06%; orden 0,982/0,989 | confirmación acotada; 9–10 reservados [E11–E13] |
| 09-13 `15dc4ca`; 09-30 `7d20831`,`25576ca`,`a7db233` | paper/README | preservar/acotar claims | diagramas de distintas rutas | sin run nuevo | inputs congelados y GIF | README ilustra teacher-CIF, no E1 libre | no usar GIF como validación [E19] |

`abbed25` integra pruebas desarrolladas antes del commit; no hay SHA individual inmutable para cada prueba. Los inits confirmatorios registran `379b942` en su linaje; el config E1 no fija inequívocamente el HEAD del propio entrenamiento [E12].

## 4. Experimental inventory

### 4.1 Familias y resultados

`ds1`=dataset1; `ds2`=dataset2. Fracciones salvo `%`, chrF/BLEU. No se recuperó test independiente de v113–v122. Cada fila diferencia ejecución, configuración y resultado; UNKNOWN no debe completarse con defaults del HEAD.

| ID / hipótesis | Arquitectura y pesos | Datos / ejecución | Train, validation y test | Conclusión respaldada |
|---|---|---|---|---|
| HIST-emb: aproximar Llama | Transformer/MSE, tabla preentrenada | notebook 2025; N/split reales UNKNOWN | todas UNKNOWN | DOCUMENTED prototipo incompleto [E01] |
| HIST-NCE: alineación global | CLIP loss, proyecciones y τ=0,07 | octubre 2025; dataset/splits UNKNOWN | UNKNOWN | implementado; no run trazable [E02] |
| HIST-soft/pad: controlar matching y fin de secuencia | TokenSoftAlignLoss y PadAwareTokenLoss, embeddings normalizados | commit c200305, octubre 2025; ejecución UNKNOWN | métricas libres UNKNOWN | implementados; no prueba de traducción [E02] |
| HIST-trainlog/model71–72 | encoder MLP/Conv/Transformer, proyección3072; sin STGCN en log | train.log:10191/2547; fecha/commit/dataset UNKNOWN | train loss1,71144 ep0; LR7e-7; DBs con trials | ejecución parcial, no resultado comparable con E1 [E20] |
| SignAlign: embedding seq2seq | STGCN/BiLSTM2 bidir + decoder Transformer propio | junio 2026; driver/checkpoint UNKNOWN | UNKNOWN | implementación parcial, no resultado E1 [E03] |
| v113: regresión | Imitator trainable; tabla 2048 fija | ds2 600 reportados; 30 ep | val loss1,449 ep24 y effrank 1,78 documentados; escalado defectuoso; test UNKNOWN | TB confirma ejecución, no loss limpia [E04, E06] |
| v114: gaussianizar | v113+SIGReg λ1 | mismo régimen; 30 ep | effrank 4,03; val 1,516 ep26; test UNKNOWN | SIGReg activo; rango no prueba grounding [E04, E06] |
| v115: CE | 20 queries→dot-product tabla | ds2 600; 30 ep reportados | top1 10,15% en detalle, ~10,9% en resúmenes; CE corrupta | no decoder causal; runaway documentado [E04] |
| v115.1: cosine/temp/LS | pred/tabla normalizados, logscale2,659; LS0,1 | ds2; 30 ep, batch 64 reportado | top1 10,87%, top5 23,20%; CE0,568 **inválida** | generación no medida [E04] |
| v116: prefix→LM | encoder/adapter ~74,31 M entrenables; Gemma3 n4 bit frozen | ds2 481/119; K20; 25–30 ep | val top1 ~28–30%; chrF20,671 ep10 en TB; test UNKNOWN | intento de frases reales, sin éxito visual demostrado [E05, E06] |
| v116.1: SIGReg | mismo modelo; λ0,1 | ds2; 25 ep | SIGReg0,02870–0,02906; chrF máximo observado19,150 ep15 entre4 registros | no demostrar equivalencia estadística con v116 [E06] |
| v117: InfoNCE auxiliar | pools prefix/text2048; τ0,07, λ0,1 | ds2; 25 ep; subbatch1 | NCE 0 en 25/25; chrF20,601; shuffled20,857 ep10 | **no probó contraste activo** [E05, E06] |
| v118 run1: CLIP | encoder + heads2048→256 entrenables; tabla fija | 600; 481/119; 38 registros | val R@1 0,025210 ep17; test UNKNOWN | contraste activo, no generación [E07] |
| v118 b run2: augment | mismo + crop/noise | 600; 37 ep | R@1 0,016807 ep11 | sin ganancia observada [E07] |
| v118c run3: VICReg | dos heads; texto de entrada estático | 600; ~29 ep | R@1 0,025210 ep3; menor divergencia de loss | no mejora de retrieval [E07] |
| v118c run4: quitar cap250 | igual; maxframes1024 | reextracción228/600 | R@1 0,029412 ep24; R@10 0,142857 ep29; train R@1 0,007277/481 | ~3,5×azar en train/val, no igualdad absoluta [E07] |
| v118d-S/T: capacidad | 14,8 M/7,5 M; dropout también cambia | configs run5/6 | UNKNOWN: sin TB5/6 recuperado | configuraciones, no ejecuciones demostradas [E07] |
| v118e run7: texto contextual | último hidden token Gemma cacheado | 600; 20 ep | R@1 0,025210 ep6; R@10 0,151261 ep16 | TB VERIFIED; no rompe techo [E07] |
| v118f run8: ampliar N | mismo VICReg/contextual | 5600; 4481/1119; 16 registros | R@1 0,002681 ep2; R@10 0,018767 ep4; azar0,000894 | no comparar % crudo con pool119 [E07] |
| v119 a subset | STGCN/RoPE + CTC palabras | 1000; documentos801/199 y 800/200; 41 ep | train 32,3→1,26; valWER~100%; test UNKNOWN | conteos contradictorios [E08] |
| v119 a full | mismo primer encoder | 5600; 40 ep | train 26→1,87; valWER99,5% ep27 | no demuestra que más datos nunca ayuden [E08] |
| v119 reemplazo | GCN3/TCN-TLP×2/BiLSTM2×256, classifier compartido | 5600; 40 ep; sin filtros | train 33,9→10,2; valWER99,25% ep15→100,23% | arquitectura distinta al primer v119 [E08] |
| v119 filtros/aug | reemplazo; 56 excluidos; aug×5; luego temporal-drop/dual decode | detenido ep12; otros smoke99/50 | valWER99,7% ep2→104,9% ep12 documentado | diagnóstico parcial, no causa única [E08] |
| v120 | classifier64, BN, ~0,35 M | ds1 3200:2560/640 | val 2,8% tras fix inicial; mismo ckpt: 88,4% stats por muestra vs1,4% eval | BN sustituye hipótesis de sobreajuste [E09] |
| v121 | GCN32/64/128, hidden 128, GN16, mean-pool,64 classes | 80 ep; seeds23/42/101 | train 98,13/97,27/98,87%; val 93,75/94,22/95,63%; media94,53%; test UNKNOWN | VERIFIED summaries; signers vistos [E09] |
| v122 | v121; 240 frames→64; trim opcional | 3200; seed 23 | val 93,59% sin trim; 91,41% trim; 90,47% trim/downsample | no mejora [E09] |
| v123/v124 | torso augment; motion/TCN/attention/dropout | configs; v124 smoke documentado | full runs UNKNOWN | pipeline detenido no es resultado negativo propio [E09] |
| v125 integridad/PCA | sin encoder nuevo; PCA 512/whitening/inversa | ds2 5600→5138; 4112/513/513 | cosine reconstrucción media 0,77034; percentil 5: 0,70355 | PCA rechazado [E15, E16] |
| v125 oracle | Gemma frozen con IDs/embeddings **GT** y PLE | 100 val; 3 ejemplos train | chrF92,86485/BLEU78,65676/RL0,98889/F10,98523 en 3 rutas | interfaz textual, no grounding [E16] |
| v126 teacher temporal | encoder transferido + CIF oracle + heads | concat2–8; 1024 train/epoch,256 val; 30 ep | teacher~0,759; free peor; README cita top1 0,369/exact 0,074 de otro régimen | no importar cifras a E1 [E10, E19] |
| v126 Optuna: calibración | alpha/quantity/freeze/schedule explorados | estudio Jun24–25:21 COMPLETE,9 PRUNED; 30 trials | mejor score 1,487591, trial 23; no es accuracy; usa oracle-length | búsqueda ejecutada, condicionada a score compuesto [E20] |
| A1/A2 token aislado | encoder/CIF; 1 clip→IDs glosa | ds1 split2560/640 | A2 oracle exact 0,823; raw0,592; A1 robustness teacher0,817 | oracle no es despliegue [E10] |
| A3 longitud | classifier longitud + rescale | single-clip; 30 ep | val deploy exact 0,818/top1 0,869 ep25; boundaryMAE28,95 | solo signers vistos [E10] |
| A3 factorial | head linear/contextual ×length mean/attention ×LS0/0,1 ×3 seeds |24 runs,15 ep, warmstart; **min=maxclips1** | contextual exact3+0,853–0,880; linear0,589–0,613 | contextual consistente; otros efectos mixtos [E17] |
| A4 scheduled alpha | mezcla0,02→0,10; rerun empieza ep5 |2 runs15 ep | ganador ep0 antes de mezcla: exact 0,900; ep14 raw0,7891/deploy 0,8891/count 0,9141 | no atribuir ganancia a mixing [E10] |
| LOSO contaminado/limpio | cadena con/sin exclusión upstream |10 folds original; 6 limpios iniciales |0,9053 anulado; limpio exact 0,0037/count 0,0094 | exact antiguo realinea al target [E10] |
| CIF rounded/affine | cantidad continua calibrada en inner | muestras pareadas con AR; folds 4–8 | exact 0,02344/0,04911/0,02567/0,01786/0,02679 | ya ejecutado, no propuesta nueva [E12] |
| AR-base | encoder CIF; decoder128×2 propio, sin PE, vocab262400, LS0,1 | concat2–8; folds 4–6; 15 ep | outer exact 0,02232/0,02344/0,01786 | generación libre D, no traducción E [E11, E12] |
| E1 | PE/vocab121/LS0/30 ep/select edit | dev4–6; confirm7–8; seed 23 | train/inner/outer en §5 | generación libre verificada; pesos ausentes [E12] |
| E2/JTA | E1+CTC λ0,3:0,7 CE+0,3 CTC | solo fold 4 demostrado | outer0,17857/edit0,61415 vsE1 0,18080/0,62952 | sin ganancia; CTCgreedy blank DOCUMENTED [E11, E12] |
| E3 | E1+unfreezeSTGCN ep10, lr 1e-5 | solo fold 4 demostrado | outer0,19531/edit0,64814; inner peor queE1 | exploratorio, no seleccionar por outer [E11, E12] |

### 4.2 Hiperparámetros e inicialización

| Familia | Optimización y presupuesto | Inicialización / límites de evidencia |
|---|---|---|
| 2025 | notebook: Adam lr 2e-4, batch 8,100 ep propuestos; MSE/cos/contraste según commit | tabla Llama-3.2-3 B; ejecución real UNKNOWN [E01, E02] |
| v113–115.1 | config heredada lr 4,5e-4, wd 0,08, clip 3,9; 30 ep reportados; seed 23; batch 64 reportado v115.1 | snapshot efectivo del optimizador y continuación por run UNKNOWN; bug altera LR efectivo/métricas [E04] |
| v116–117 | CE-AR, K20, batch 32 nominal/subbatch1, seed 23; 30/25/30 ep máximas; 25 observadas v117; 50 newtokens | GemmaE2 B4 bit frozen; encoder/adapter trainable; `check_checkpoint` permite reanudar: init limpia por run UNKNOWN; LR efectivo UNKNOWN [E05] |
| v118 | AdamW3e-4, wd 0,05, warmup5; batch 16 base/32 VICReg; seed 23; 60/80 ep máximas,20e/f; patience20/25 | heads256, texto2048 cacheado; encoder grande1024×6, FF2816; frames 768 base/1024 c–f; procedencia exacta de cada init no totalmente registrada [E07] |
| v119 reemplazo | AdamW4e-4, wd 1e-4, batch 16,40 ep, decay÷10ep20/35, seed 23, maxframes1024 | sin LM preentrenado; vocab construido solo train; blank0/unk1; motion config no prueba ejecución [E08] |
| v121/122 | AdamW1e-3, wd 1e-4, batch 16,80 ep, LS0,1; seeds23/42/101 v121,23 v122 | clasificación64; val 10 porclase; v122 H5 nuevo con metadata [E09] |
| v126 factorial | AdamW3e-4, wd 1e-4, batch 4,2048 train/epoch,512 val,15 ep; seeds23/42/101, splitseed23; αloss5, qty1, emb0,05, length1 | resume Etapa1; 43 weights cargados,2 omitidos,20 missing registrados; freeze porfase; **un clip** [E17] |
| E1 | AdamWdecoder3e-4, visual3e-5 desdeep5, wd 1e-4; batch 2×accum2; 30 ep,2048 train/epoch,512 inner,896 outer; seed 23 | encoder completo y classifier/bias desde CIF limpio por fold; resto decoder nuevo; STGCN/spatial frozen; 32 content+EOS; PEhasta4096 frames [E11, E12] |

## 5. Seq2seq investigation

### 5.1 Distinción de tareas

| Categoría | Evidencia | Conclusión permitida |
|---|---|---|
| A. Reconocimiento de glosas | v120–122,64 logits | clasificación aislada; 94,53% no es traducción [E09] |
| B. Concatenación/tokenización de glosas | generador sintético | targets derivados de etiquetas españolas, sin gramática natural [E13] |
| C. Embeddings lingüísticos | v113–114, SignAlign, head auxiliar v126 | aproximación vectorial, no generación demostrada [E03, E04] |
| D. Tokens autoregresivos | GemmaBridge v116–117; VideoTokenDecoder | sí hay intentos y evaluaciones libres [E05, E12] |
| E. Video continuo→oración | intentos ds2 v116/117; CTC de transcripciones v119 | no éxito visual demostrado; E1 no evalúa E |
| F. Corrección posterior LLM | oracle v125, tolerance sweeps | formatear texto conocido; no certifica recuperar errores visuales [E16, E19] |

### 5.2 v115–v115.1: no autoregresivos

Imitator usa STGCN→TransformerRoPE→20 queries→cross-attention a frames→proyección2048. Cada query predice una posición sin consumir tokens anteriores. La loss recorta al mínimo de longitudes pred/target y excluye padding. Llamar a su accuracy «teacher-forced» confunde: hay evaluación condicionada a posiciones/longitud target, pero no historia textual GT alimentada al modelo. No hay contrato BOS/EOS ni stopping aprendido. v115 usó dot-product; el HEAD normalizado pertenece al cambio v115.1 [E04].

### 5.3 SignAlign: seq2seq de embeddings sin ejecución demostrada

`d1c95a4` añade `KeypointEncoder`, `AlignmentDecoder`, `SignAlignGCN_KeypointsOnly`; permanecen en `imitator.py:128–306`. Encoder STGCN/BiLSTM bidireccional2 capas; memoria proyectada; decoder Transformer propio con PE sinusoidal y máscara triangular. Entrena con BOS vectorial aprendido y embeddings target desplazados. Infiere realimentando embeddings predichos durante50 pasos, sin EOS. No hay resultados de texto generado ni checkpoint atribuible. La construcción STGCN usa una firma diferente de la de Imitator, señal de prototipo no integrado. **DOCUMENTED implementación parcial**, no evidencia de arquitectura entrenada [E03].

### 5.4 v116–117: sí hubo intentos de frases reales con Gemma

Visual: STGCN/RoPE,20 queries; adapter LayerNorm+Linear 2048, normalización L2 y escala aprendible inicial√2048. Decoder: **Gemma-3 n E2 B instruction-tuned, preentrenado,4 bit, congelado**. Conexión mediante concatenación de prefix continuo y embeddings de texto, no nueva cross-attention dentro de Gemma. Encoder/queries/adapter/escala entrenables; ~74,31 M parámetros documentados. No se verificó QLoRA ejecutado en esta línea [E05].

Teacher forcing usa `target[:,:-1]`; labels prefix0…K−2=-100 y última posición prefix predice primer token textual. Padding textual excluido de attention; causalidad del LM. `be8e9f2` corrige primer-token supervision, padding y norma del prefix. El loader del bridge fija max_seq_length128; greedy limita50 newtokens. La config histórica de datos permite4000 frames, sin prueba de cobertura efectiva en cada run. **K20 no limita la salida AR a20**. BOS/EOS dependían del tokenizer/bridge de esa versión, sin contrato equivalente al EOS explícito de E1; no atribuirles automáticamente la política v125 posterior [E05, E16].

Se usaron videos continuos segmentados de ds2. Generación greedy sin boundaries/longitud target suministrados; chrF frente a referencias, primero cada5 ep/4 batches y después cadaépoca/valcompleta. **VERIFIED como registro:** chrF20,6707 v116 y 20,6012 v117 ep10. **DOCUMENTED:** re-eval v117 reporta retrieval@1=0/119 en seis snapshots y Wilcoxon no significativo. `/tmp/reeval_v117_gap.py` no permanece como script reproducible. Shuffle apoya escaso grounding, pero «enteramente prior lingüístico» es interpretación, no descomposición cuantitativa de señal [E05, E06].

El arreglo de `per_layer_inputs` (PLE) de Gemma3 n en v125 es posterior. Su equivalencia ID/embedding no puede extrapolarse retrospectivamente al bridge v116–117 [E16].

### 5.5 v119: CTC sobre palabras españolas

Primera versión: STGCN/TransformerRoPE→frameclassifier. Reemplazo: GCN3→TCN/TLP×2→BiLSTM2→classifier compartido, loss `CTC(Ys,Y)+CTC(Yl,Y)` y auxiliares según configuración. Targets: palabras de transcripciones españolas, vocab de train, **no glosas anotadas de la lengua de señas**. La correspondencia monótona palabra–video requerida por esta formulación no fue validada. Que el reporte lo llame CSLR no crea supervisión CSLR. GreedyCTC elimina blanks/repeticiones, sin BOS/EOS ni teacher tokens. Su WER~99–100% no evalúa un decoder AR ni refuta SLT seq2seq [E08].

### 5.6 v125: oracle textual

100 ejemplos val; entrada target mediante IDs exactos/embeddings exactos/inversa de transformación; 3 ejemplos train fijan formato. chrF92,86485 en 3 rutas. Responde «¿puede transportarse texto conocido por este bridge?», no «¿puede generarse desde video?». El prompt se seleccionó con sweep en val: desarrollo, no test final independiente. PCA 512 no pasó fidelidad; full_standardized mantiene2048 D [E16].

### 5.7 v126/CIF: teacher temporal y longitud oracle

Encoder distinto del Imitator antiguo: STGCN3 canales32/64/128, GN, proyección128, TCN2×k3, Transformer1 capa/4 heads, originalmente sin PE. CIF integra masas α y emite vectores; head contextual con posición y self-attention **no causal** entre slots. Heads auxiliares2048 y longitud; no LM en entrenamiento. Produce IDs paralelos, no autoregresión [E10, E17].

- `teacher_alpha`: usa boundaries y token spans GT.
- `pred_rescaled_to_target_len`: alpha predicha, longitud GT.
- `pred_raw`: no fuerza longitud.
- `pred_rescaled_to_pred_len`: longitud estimada por classifier.
- `pred_rescaled_to_rounded_count`: cantidad continua redondeada, con calibración opcional.

Los dos primeros son oracles. «Teacher forcing» aquí suministra segmentación, no palabras previas. Muchos A1–A4 y los24 runs factoriales usan **un solo clip**; el scaffold inicialv126 sí concatena2–8 [E10, E13, E17].

### 5.8 VideoTokenDecoder/E1: generación libre

BOS=2, EOS=106, PAD=0; 32 tokens de contenido+EOS. Son los IDs fijados en este protocolo; no asumir que106 es el EOS universal de cualquier Gemma. Entrada `[BOS,Y]`, labels`[Y,EOS]`, padding excluido, máscaras causal y de memoria por longitud **visual**. Greedy comienza solo de BOS, prohíbe EOS en primer paso, termina con EOS/máximo33 y remapea dense→Gemma. No acepta count/boundaries/targetlength. Targets se usan solo para métricas en evaluación [E12].

Decoder Transformer propio:128 D,2 capas,4 heads, FF512, dropout0,1, embedding/salida atados. **Emitir IDs Gemma no implica usar el decoder preentrenado de Gemma.** Su tabla 128 D se inicializa desde classifierCIF, no tabla input Gemma2048. E1 restringe121 IDs:118 content+3 specials. No es open-vocabularySLT. Encoder desde CIF limpio por fold; STGCN/spatial frozen; TCN/Transformer se abren ep5. E3 abreSTGCN ep10 [E12].

Datos:2–8 clips elegidos con reemplazo del pool correspondiente, orden aleatorio, gaps 0–8 ceros después de normalizar. Boundaries existen para construcción/diagnóstico, **no entran al AR**. No hubo scheduled sampling de tokens en E1; el scheduledα de A4 es otro mecanismo [E10, E12, E13].

| Fold / rol | Mejor época inner | Train CE en esa época | Inner strict / edit | Outer strict | Outer edit | Length MAE | W/L/T vs CIF |
|---|---:|---:|---|---:|---:|---:|---|
|4 desarrollo|28|0,05021|0,19141 / 0,61462|0,18080|0,62952|1,23214|144/3/749|
|5 desarrollo|24|0,07090|0,18945 / 0,61363|0,19754|0,61849|1,02567|140/7/749|
|6 desarrollo|24|0,05911|0,12695 / 0,55676|0,16853|0,58503|1,08705|131/3/762|
|7 confirmación|28|0,05675|0,15625 / 0,59724|0,12500|0,53017|1,36272|99/3/794|
|8 confirmación|27|0,05261|0,07813 / 0,45080|0,14063|0,58737|1,48772|108/6/782|

**VERIFIED:** `metrics.jsonl`/`outer_test.json` [E12]. Mejor época reconstruida por `(inner edit,inner strict)`. Train CE es teacher-forced, no exact libre. Sin binario no se cotejó el estado del checkpoint cerrado.

Confirmación: IC95% guardados de strict, bootstrap por secuencias: fold 7 [0,10379; 0,14732], fold 8 [0,11830; 0,16406]. Orden pareado entre glosas identificadas 0,98154/0,98892; no significa 99% de todas las glosas correctas. Precision/recall 0,62463/0,53742 y 0,66424/0,58595. Con 8 señas, strict 0/109 en ambos folds; con 2 señas 0,39831/0,51695. Falla residual en identidad, omisiones y acumulación con longitud [E12].

**Límite causal:** E1 cambia PE, vocab, LS, épocas y selección simultáneamente. Permutación demuestra sensibilidad al orden del modelo final; **no aísla cuánto de la ganancia proviene de PE**. La afirmación histórica «causa raíz confirmada» requiere esta salvedad [E11, E12].

## 6. Regularization investigation

### 6.1 Pérdidas y tensores

Sea `valid` el conjunto no padding y `sg` stop-gradient.

| Estrategia | Función exacta / geometría | Tensor / representación textual | Evidencia de resultado |
|---|---|---|---|
| MSE+cos | `Lm=mean_valid mean_D((z−e)^2)`;`Lc=mean_valid(1−cos(z,e))`;`L=Lm+sg(Lm/(Lc+1e−6))*Lc` | tokens 2048 en serie Gemma; input embeddings estáticos; recorte a longitud mínima | v113 colapso; beta adapta magnitud, no λ fijo; bug limita comparabilidad [E04] |
| CE-vocab | v115`z Eᵀ`; v115.1`exp(logs)*norm(z)norm(E)ᵀ`; CE con LS0,1 | queries por posición vs tabla 262400×2048 fija | normalización elimina runaway por norma; no certifica grounding; escala final 15,9 DOCUMENTED [E04] |
| SIGReg/Epps–Pulley | estandariza dimensiones; proyecta U unitarias;`mean_s Σk wk[(mean cos(tkXU)−exp(−tk²/2))²+(mean sin(tkXU))²]` | tokens válidos aplanados; v114 regresión/v116.1 prefix B×20×2048; sin texto target | λ1 v114/0,1 v116.1; 1024 slices,17 nodos Gauss–Hermite, remuestreo [E06] |
| InfoNCE v117 | `S=norm(meanKprefix)norm(masked_mean(text))ᵀ/0.07`;`L=.5(CE(S,I)+CE(Sᵀ,I))`; total CE-AR+0,1 NCE | global 2048; texto input Gemma estático | NCE 0 en 25/25; B=1 carece de negativos [E05, E06] |
| CLIP v118 | CE simétrica;`S=exp(s).clamp(max100)*v tᵀ`; τinit0,07 aprendible | pools globales, heads LN→Linear 256→GELU→Linear 256; input texto cacheado | activo; R@1 base 0,02521, aug0,01681 [E07] |
| VICReg v118 | `25*MSE(v,t)+25*(relu(1−sqrt(var(v)+1e−4)).mean+ídem(t))+Σoffdiag(Cov(v))²/D+ídem(t)` | vectores 256 **sin normalizar L2**; texto estático c/d, contextual e/f | controla divergencia de loss, no mejora retrieval de forma suficiente [E07] |
| PCA/whitening v125 | PCA 512 ajustado train, whitening/inversa; gates cos mean≥0,99, p5≥0,97 | tabla 2048→512→2048; 8256 trainIDs/525 valIDs en artefacto |0,77034/0,70355: rechazado; full_standardized [E16] |
| Auxiliares v126 | `CEtokens+.05*SmoothL1(emb,table)+CElen+5*Lalpha+L1(sumα,Lgt)` en factorial; primeras3 ep sin CEtokens/emb | frameα, slots 128→2048 y targets estáticos; `logit_l1` en run oficial | warmup conserva classifier; no resuelve longitud LOSO [E10, E17] |
| CTC dual | `CTC(logp_short,Y)+CTC(logp_long,Y)` más auxiliares según config | palabras ds2 y blank, sin glosas alineadas GT | WER alto; supuesto de monotonía no certificado [E08] |
| E1/E2 | E1 CE causal LS0; E2`0.7CE+0.3CTC` | IDs restringidos; head CTC128→122 incluido blank | E2 sin ganancia fold 4; no 3 seeds [E11, E12] |

SIGReg aplanando tokens puede ser no nulo con B=1 porque hay 20 slots. Eso regulariza diversidad **entre slots**, no necesariamente entre videos. Estandarizar elimina media/escala marginal antes del test; estadístico bajo no prueba isotropía de salidas originales. `0,1×0,029/4≈0,000725` muestra pequeña contribución escalar, **no relación de normas de gradientes** ni necesidad demostrada de λ≥10 [E06].

Los objetivos históricos de octubre de 2025 también deben conservarse. `TokenSoftAlignLoss` normaliza predicciones y texto, calcula `A=softmax(5*P Eᵀ)` dentro de cada par y minimiza `mean(1−cos(P,A E))`; regularización de entropía opcional, default 0. No introduce negativos entre videos ni genera tokens. `PadAwareTokenLoss` combina 0,2×alineación coseno no-pad + 0,5×(pull-pad + push-no-pad) + 0,2×monotonía de similitud a PAD + 0,2×margen de frontera; márgenes push 0,2/boundary 0,3. Su ejecución y efecto son UNKNOWN, aunque el código existe en `c200305` [E02].

Para CTC/TLP, el HEAD suma además `0.1*L_u+0.1*L_p`: separa frames pares/impares, `d=x_o−predictor(x_e)`, `s=x_e+updater(d)`, `L_p=mean(d²)`, `L_u=mean((s−x_o)²)`, sumados entre dos TLP. No son pérdidas contrastivas ni geodésicas (`components/tlp.py:32–58`; trainer CTC:147–149,292–294) [E08].

En v126, `Lalpha(current)` es SmoothL1 entre alpha predicha/target, ponderada por `1+20*1[target>0]`; `logit_l1` aplica la misma SmoothL1 a logits frente a `logit(clamp(target,1e−4,1−1e−4))`; `kl_quantity` normaliza masas a distribuciones y aplica KL; `qty_only` pone esta loss a cero. Quantity L1 sigue siendo independiente. El nombre `logit_l1` no significa L1 puro (`train_temporal_v126.py:428–461`) [E10].

### 6.2 Diagnósticos y selección

- TB de v113/v114 existe; el detalle v115.1 registra effrank 1,78/4,03 y cos 0,847/0,796. «Sin métricas» en un resumen posterior es falso; los pesos sí fueron eliminados [E04, E06].
- **DOCUMENTED:** promedio de embeddings Gemma input: cos entre frases~0,80, vecino cercano mediana~0,90, effective rank 123/2048. Motivó v118e. Allí se cacheó el último hidden state de última capa, contextual por causalidad; no se entrenó el LM [E07].
- **VERIFIED:** diagnóstico v118c train R@1=0,0072765, val=0,0294118; azar 1/481 y 1/119. `gap_factor≈1` compara relativo al azar. No son accuracies absolutas iguales [E07].
- Los comentarios «colapso imposible por construcción» de InfoNCE/VICReg son excesivos: penalización no garantiza optimización exitosa. Además retrieval usa comparación estricta `>` para rango: empates reciben rango 1. Una solución constante puede inflar esta métrica; no se demostró que explique específicamente los runs observados [E07].
- Baselines: unigrama v115.1; shuffleprefixv117; azar por poolv118; oracle alpha/longitudCIF; factorial con headlinear; AR-base y CIFaffine pareado. τ0,07 y VICReg25/25/1 son valores configurados por precedente; no búsqueda empírica probada. Optuna v126 sí conserva una búsqueda ejecutada: véase el detalle de score y trials en §13.2; no certifica un óptimo global [E04, E07, E10, E17].

### 6.3 Geometría hiperbólica

**VERIFIED búsqueda acotada:** `git grep -i -E 'poincar|lorentz|hyperbolic|riemannian|geodesic'` en 54 tips, archivos`.py/.toml/.yaml`:cero coincidencias. Inspección adicional de fuentes de celdas locales:sin implementación identificada. No se encontró ensayo ejecutado, loss geodésica, Poincaré, Lorentz o RiemannianAdam en Imitator. Esto no prueba ausencia en repositorios borrados, commits inalcanzables o máquinas externas. Cualquier ensayo no conservado es **UNKNOWN** [E18].

## 7. Dataset and supervision audit

### 7.1 Procedencia y función

| Dataset | Tamaño / naturaleza | Supervisión y uso | Lengua/procedencia |
|---|---|---|---|
|dataset1|3200 videos locales contados; 64 clases×10 signers×5 reps documentado|clips aislados/glosas españolas; v120–122 y base v126/E1|paper actual identifica LSA64 y Ronchetti2016; auditoría v125 decía UNKNOWN. Atribución actual DOCUMENTED, no checksum del distribuidor [E09, E14, E19]|
|dataset2|8459 videos locales contados; 5600 HDF5 histórico; 5138 keptv125|frases españolas/programas noticiosos; v113–119/oracle125|LSA por nombres de programas argentinos; canal CN Sordos documentado; no LSP como reportes antiguos [E14, E15]|
|dataset3|38 clips EDA|38 glosas aisladas; run no demostrado|español es idioma del texto; lengua de señas UNKNOWN [E14]|
|dataset4|4 videos/1 metadata EDA; mediana 46 s, máx 104 s|narración205 palabras; excluido/qualitativo en config|nombres LSP y documentación; identificación externa UNKNOWN [E14]|
|dataset5|359 videos/354 metadata,214 labels|aislado con outliers; máx 167,77 s; sin run probado|lengua UNKNOWN [E14]|
|dataset6|259 videos EDA; mediana 143,658 s, p95 598,672 s, máx 4108,415 s|narraciones/transcripciones, ruido de alfabetos; secundario en config115, no transferencia ejecutada demostrada|LSP DOCUMENTED; raw retirado según cleanup [E14, E15]|
|dataset7|526 videos EDA,144 labels|aislado; sin entrenamiento demostrado|lengua UNKNOWN; raw retirado según cleanup [E14, E15]|

Dataset1 conserva `meta.csv`/`label.csv`; ds2 conserva únicamente `videos/`, sin CSV/subtítulos/segmentación originales. El notebook local ds1 no aportó URL de origen. Labels españolas no identifican la lengua de señas. La atribución posteriorLSA64 mejora la documentación, pero falta trazabilidad de los archivos respecto al distribuidor [E14, E19].

### 7.2 Extracción, articuladores y límites

`build_dataset2_h5.py:47–90` usa RTMPose/rtmlib, to_openpose y elige persona por energía de muñecas. Constante efectiva 134 puntos, docstring 137, paper 133: no asumir layouts idénticos. `remove_keypoints:142–150` retiene 7 pose + 64 face + 20 por hand=111. Tensor 2 D(x, y), sin confianza utilizada. Elegir por índice de persona sin reidentificación estable introduce riesgo de cambios de tracking: **INFERRED**, no error medido aquí [E14].

Normalización: abs coordenadas; inválidos si ambas<5; extremos de puntos válidos por clip, min–max por eje, conservando marcadores inválidos. Los gaps cero se añaden después. Usa estadísticas del clip completo; no equivale a pose3 D ni normalización causal por frame [E13, E14].

Historia ds2: cap250 en primerbuild600; 228 truncados documentados; reextracción hasta 1400; trainingv118 base cap768, c–fcap1024. «Reextraído sin 250» no garantiza que el trainer consumiese todo el video. Al ampliar600→5600, las claves numéricas no preservaron identidad; `0..599` no identifica el subconjunto original sin `video_id` [E07, E14].

Ds1 H5 original T≤150,134×2 reportado; v122 T≤240,3200/3200, metadata signer/repetition. El remuestreo a 64 pertenece a la ablación v122, no al generador base E1. E1 concatena hasta 8 clips y buffer PE 4096 frames; 32 BPE tokens no significa 32 señas [E09, E12, E13].

### 7.3 Auditoría de integridad v125

**VERIFIED artefactos:** total 5600, kept 5138, excluidos 462; primera causa:truncated 413, frames/token 16, min_tokens 29, min_frames 4. Son causas prioritarias, no prevalencia total de anomalías. Reglas:min 8 frames, min 2 tokens, min 2 frames/token; keypoints/label/embeddings/IDs presentes; filas ID/embedding iguales; sin NaN/Inf. Truncado si framecountH5/video<0,9; video no abierto o framecount 0 también se marca truncado [E15].

Split seed 23:4112/513/513. **Recomprobado:** IDs disjuntos y unión igual a kept. El algoritmo agrupa por `source_group`=columna`video`del CSV. El JSON solo conserva IDs; sin leer HDF5 no se revalidó disjunción de programas, y sin CSV no se certificó calidad de esa metadata. Signer inferido ds2 no es identidad estable fiable entre programas [E15].

**VERIFIED bytes:** siete artefactos v125 coinciden en tamaño/SHA con el manifest congelado, incluidoHDF5 de2.244.045.489 bytes y tabla Gemma de1.074.792.061 bytes. Existencia/hash no equivale a nueva inspección semántica del HDF5: python actual carece de h5py; el intérprete histórico Sign-env no existe en su ruta. No instalé dependencias. El contenido se reconstruye mediante artefactos históricos, no relectura de todos sus arrays [E15, E16].

### 7.4 ¿Dataset2 permite entrenar traducción continua confiable?

**No está suficientemente certificado para un experimento concluyente de SLT. Tampoco está probado que sea intrínsecamente inutilizable.**

1. **DOCUMENTED:** interpretación continua LSA y frases españolas. EDA:mediana 7,407 s, p95 24,228 s, máx 89,833 s; mediana 13 palabras, p95 37, máx 153; 8016 labels únicas/8459 [E14].
2. **VERIFIED:** existe integridad computacional mejorada,5138 clips seleccionados y videos fuente. El filtro no comprueba cobertura semántica, desfase intérprete–subtítulo ni correspondencia de oración completa [E15].
3. **DOCUMENTED no comprobado:** configs mencionan 3088 clips desalineados>1 s. No se recuperó listado causal/denominador/auditoría reproducible que permita reconstruirlo. No usarlo como fracción verificada de ds2. Posible desfase entre subtítulos y LSA es riesgo plausible, no offset medido aquí [E14, E15].
4. **INFERRED:** CTC exige monotonía entre frames y palabras españolas; LSA/español no garantizan mismo orden. Fallar con ese target no descarta un seq2seq adecuado.
5. «96%labels únicas/93%singleton»en 5600 es DOCUMENTED, pero **no implica inviabilidad de SLT**. Seq2seq comparte tokens/patrones entre frases únicas; CTC también comparte palabras. Los reportes que exigen frases repetidas como condición necesaria exageran la evidencia [E08, E09].
6. Es necesario restaurar metadata/subtítulos y certificar una muestra por programa/duración con competencia en LSA, anotando cobertura, offset, signer confidence y source group. Sin eso, otro resultado negativo mezcla arquitectura, supervisión y sincronía.

## 8. Failure analysis

| Síntoma | Evidencia | Hipótesis causal y certeza | Intervención | Resultado posterior | Estado |
|---|---|---|---|---|---|
|embeddings poco discriminativos|effrank 1,78/4,03|media condicional de regresión:INFERRED|SIGReg/CE/contraste|rango mejor, SLT no demostrada|regularizar no bastó [E04, E06]|
|CE v115.1 imposible|0,568 vs floor 1,248|doble división:confirmada por código/commit|divisiones out-of-place Jun15|logs corregidos coexisten con antiguos|invalidar 0,568 [E04]|
|prefix colapsa/ignora video|shuffle≥real; effrank cae|prior LM+TF:INFERRED|SIGReg/InfoNCE|SIGReg pequeño activo; NCE 0|no ensayo de NCE activo aquí [E05, E06]|
|retrieval v118 pobre|R@1~2–3%; train relativo azar similar|formulación global/capacidad/datos:no aislados|VICReg, ctx, ampliar N|e: 2,52%; f: 0,268%/1119|objetivos insuficientes en ese régimen [E07]|
|frases cortadas|228/600 cap250 DOCUMENTED|supervisión parcial:INFERRED|reextraer 1400|R@1: 0,0252→0,0294; cap de train 1024 persiste|no ganancia causal pura sin réplicas [E07]|
|CTC blank|WER~100%, posteriors/activaciones|BiLSTM reduce variación; no prueba pérdida de toda información|TCN/TLP/dualCTC/aug/filtros|persiste; summary actual 5 clips no 300|causa única UNKNOWN [E08]|
|v120 azar en eval|mismo ckpt: 88,4%stats por muestra/1,4%running stats|BN con B=1:fuerte apoyo por intervención|GroupNorm|v121: 94,53±0,80%val|resuelto para signers vistos [E09]|
|trim v122 degrada|91,41/90,47 vs93,59%|pierde postura/contexto:INFERRED|staged gate|full v123/124 no continúan|no fracaso propio de esas variantes [E09]|
|learnedCIF cero fires|α logit p50−26,49 ep3|mezcla 50% demasiado pronto:apoyo diagnóstico|warmup target-only|raw top1 hasta 0,223; boundary 29,46|no alineación natural resuelta [E10]|
|LOSO aparente 90%|test signer visto upstream|leakage DOCUMENTED|cadena limpia por fold|exact 0,0037/count 0,0094|resultado previo anulado [E10]|
|lengthhead falla unseen|oracle length~0,59 exact vs deploy~0,004|classifier buckets/generalización:INFERRED|round/bias/affine|comparadores exact1,8–4,9%|ya probado [E10, E12]|
|ARbase insensible a orden|86,3%pred invariantes DOCUMENTED|sin PE global plausible; TCN/GCN sí dan orden local|paquete E1|free exact 12,5–19,8%; shuffle degrada|paquete eficaz, PE aislado no probado [E11, E12]|
|E1 falla largos|8 señas0/109 ambos folds; recall 0,54/0,59|identidad/omisiones acumulativas:INFERRED|E3 unfreeze fold 4|outer mejor / inner peor|exploratorio, no adoptar por test [E12]|
|disco lleno|ENOSPC Jun20 DOCUMENTED|pesos ocupaban 43,7 GB|borrar 113–117|métricas sobreviven, sin reinferencia|restaurar backups [E15]|
|bridge/runtime|4 bit HF sin init, Torch/Triton, PLE|plumbing concreto documentado|Unsloth, eager, PLE oracle|equivalencia v125|no certifica versión v116 retroactiva [E05, E16]|

Las afirmaciones «5,6×datos refuta falta de datos», «filtros refutan ruido» o «varianza identifica causalmente el backend» exceden el diseño. Escalar la misma colección, filtrar 1% y medir variación no aísla esas hipótesis. El reporte CTC incluso menciona~480–580 train junto a runs 4481 train; no usar esa cantidad para atribución [E08].

## 9. Current state

### 9.1 Implementación, entrenamiento y evaluación

- **Implementado y con artefactos de ejecución:** v121 GN, v118 CLIP/VICReg, v119 CTC, v126 CIF, VideoTokenDecoder/E1. Hay tests y reportes de pasadas históricas; no se reejecutaron sin Torch [E07–E12, E17].
- **Resultados recomputables sin inferencia:** predicciones E1 outer/robustness, CIF comparadores, summary v121 y factorial hasheado. No equivale a reproducibilidad de entrenamiento ni determinismo GPU garantizado [E09, E12, E17].
- **Sin full run verificado:** v118d-S/T, v123/v124 completos; SignAlign parcial; E3 solo fold 4, no 5/6; frameprobe implementado sin ejecución final recuperada [E03, E07, E09, E11].
- **Solo propuesta histórica/no ejecutada:** scheduled sampling de tokens post v117; PrefixImitator+QLoRA antes del pivote CTC; hiperbolicidad; SLT continua exitosa. No se identificó mT5 implementado en Imitator [E07, E18].
- **Descartado bajo condiciones concretas:** E2 CTC fold 4, trim v122, mixing α A4 para deploy. No son refutaciones generales de CTC, segmentación o contraste [E09–E12].

### 9.2 Mejores resultados frente a disponibilidad de checkpoints

| Tarea | Candidato histórico | Disponibilidad actual |
|---|---|---|
|aislado 64, signers vistos|v121 seed 101 val 95,63%; media 3 seeds: 94,53%|`../outputs/checkpoints/121/101/best_top1/checkpoint.pth` referenciado, ausente|
|retrieval ds2/val 119|v118c/run4 R@1: 0,029412|`../outputs/checkpoints/118/4/best_r1/` sin binario localizado|
|CTC ds2|ninguno funcional; WER~99%|pesos ausentes|
|CIF un clip, signers vistos|A3/A4 ep0, exact~0,90 del protocolo antiguo|punteros históricos, binarios ausentes|
|AR LOSO confirmatorio|E1 fold 7/8 strict 0,125/0,140625; cada uno para su outer|`checkpoint_closed.pt` ausentes; SHA `480bbecd…` / `60763fb4…` registrados|
|SLT continua real|ningún checkpoint exitoso demostrado|UNKNOWN/sin candidato|

Búsqueda `*.pt/*.pth` en `../outputs`:**cero**. Tampoco se localizaron pesos experimentales en outputs del repo. La tabla Gemma y transform v125 en `../data/processed` no son modelos visuales. SHA guardado verifica una futura restauración, no recrea pesos [E12, E15, E16].

### 9.3 Transferencia y reutilización

Cadena reconstruida: **v121 aislado → extractor corporal/linear_hidden → v126 A1 teacher → A2 learnedCIF → A3_pre_decoder → decoder_final → promotion → VideoTokenDecoder → E1/E2/E3**. AR copia encoder completo yfilas classifier/bias. EnLOSOlimpio cadafold excluye outer desde v121; no basta fine-tunear unmodelo global visto por10 signers. Configsfold7/8 guardan padres/hashes/train 8/inner1/outer1 [E10, E12].

**Si se restauran pesos**, STGCN/GN/TCN/Transformer sirven de init visual; headsCIF como oracle/diagnóstico; AR/EOS para tokens conocidos. Splits, lineage, métricas y bootstrap son reutilizables sin pesos. ParaSLTnatural, vocab121 y 32 tokens no cubren texto abierto y elgenerador sintético no aporta coarticulación. No hay actualmente pesos visuales disponibles que permitan prometer transferencia sin volver a entrenar: se requieren backups externos [E12–E16].

## 10. Geo-Sign compatibility assessment

Consulta remota read-only el2026-10-19; HEAD`5cacfe5b6cfc560fea5af0eb48f1544882a5844a`. Se leyeronREADME,`models.py`,`fine_tuning.py`; no se entrenó ni se comprobó checkpoint. Las cifras publicadas son **DOCUMENTED**, no reproducciones [E18].

| Componente Geo-Sign | Antecedente Imitator | Diferencia relevante |
|---|---|---|
|poseSTGCN|STGCNdesde2025, v121/E1|multipartbody/hands/face ypretraining distintos|
|contraste visual/texto|hist2025, v118 CLIP/VICReg|sí hubo contraste euclidiano activo; v117 no lo prueba|
|texto estático/contextual|input Gemma vsúltimohiddenv118|Geo usa input/decoder mT5, no idéntico target|
|seq2seq preentrenado|Gemma frozen prefixv116|mT5encoder–decoder yposeinputs; sinfreezeLMglobal en código inspeccionado|
|Poincaré learnable|ninguno identificado|expmap0, distanciasgeodésicas, curvatura aprendida|
|comparación token geométrica|atención euclidiana enqueries/CIF/AR|atenciónhiperbólica porpartes/tokens ymidpoints|
|RiemannianAdam|ninguno en Imitator|separaManifoldParameters/curvatura de AdamW|

[Fuentes fijadas al commit](https://github.com/ed-fish/geo-sign/tree/5cacfe5b6cfc560fea5af0eb48f1544882a5844a), [modelo](https://github.com/ed-fish/geo-sign/blob/5cacfe5b6cfc560fea5af0eb48f1544882a5844a/models.py#L56-L103), [optimizadores](https://github.com/ed-fish/geo-sign/blob/5cacfe5b6cfc560fea5af0eb48f1544882a5844a/fine_tuning.py#L164-L206).

**VERIFIED implementación remota:** proyección `Linear(x)*exp(logscale)` con `expmap0`fp32 aPoincaré. Loss unidireccional `CE(−d(p_i,t_j)/τ+margin*1[i≠j],i)`; τsigmoid en(0,01; 2), margen aprendido. Mezcla `α*CEtranslation+(1−α)*Lgeometry`, α programado/aprendido. No es solo reemplazar coseno por otra distancia [E18].

README reportaCSL-DailytestBLEU4:Uni-Signpose25,61; euclideanpooled25,72/token25,98; hyperbolicpooled27,17/token27,42; ROUGE-Lpose54,92/hypertoken57,95. Benchmark, lengua, decoder y supervisión diferentes: no comparables conexactE1. No se verificó la ejecución de cadaablación delpaper.[Tabla oficial](https://github.com/ed-fish/geo-sign#evaluation).

**INFERRED diseño compatible:** preservar decoder E1 y supervisión, añadiendo rama auxiliar visual/textual con proyecciones emparejadas. Comparar CE solo, CE+contraste euclidiano, curvatura fija y curvatura aprendida. Igualar dimensión, normalización, pooling, τ, margen, batch negativo, peso auxiliar, presupuesto y selección. `c=.001`aprendible no es baseline plano exacto: puede alejarse de cero. Control euclidiano explícito o curvatura realmente fija [E18].

Texto GT puede supervisar la rama auxiliar durante train, pero no entrar al decoder en inferencia. Verificar gradiente auxiliar no nulo/B>1 real, magnitudes por grupo y operaciones geométricas fp32. Mantener decoder euclidiano permite probar regularización sin reescribirlo; no reproduce toda la arquitectura Geo-Sign [E05, E12, E18].

La geometría no aporta anotaciones, sincronía, pesos perdidos, diversidad de signers, EOS correctamente evaluado ni control causal PE. Tampoco añade orden global donde no exista PE. La jerarquía corporal por sí sola no demuestra ganancia por curvatura frente al fallo de identidadE1 [E11–E15, E18].

## 11. Research gaps

1. **Supervisión continua:** ¿qué fracción ds2 tiene cobertura traductiva y desfase manejable? Faltan metadata y certificación LSA [E14, E15].
2. **Atribución E1:** ¿cuánto aportan PE, vocab, LS, épocas/selección? No están aislados [E11, E12].
3. **Generalización visual:** ¿unfreeze o keypoint dropout mejora identidad LOSO? E3 solo fold 4; A1 detectó fragilidad a ausencia de articulaciones [E10, E12].
4. **Contraste auxiliar activo:** v118 probó retrieval global, no auxiliar junto a E1; v117 NCE cero no lo refuta [E05–E07].
5. **Curvatura:** no ensayo local; fallos MSE/CIF no prueban necesidad hiperbólica [E18].
6. **Reproducibilidad:** pesos e intérprete ausentes; config E1 sin SHA propio inequívoco; una seed confirmatoria [E12, E15].
7. **Unidad estadística:** secuencias sintéticas reutilizan clips; bootstrap 896 secuencias no equivale a 896 videos/signers independientes. Orden pareado depende de glosas identificables [E12, E13].

## 12. Prioritized experiments

Son propuestas, no ejecuciones. Restaurar pesos y metadatos es un prerrequisito operativo. Los folds 7–8 ya fueron observados. Reservar 9–10 para confirmar una única selección de desarrollo congelada previamente.

| Prioridad | Hipótesis | Baseline | Cambio mínimo | Dataset | Métricas | Criterio de éxito | Riesgo |
|---|---|---|---|---|---|---|---|
| 1 | Un subconjunto certificado permite interpretar un ensayo de SLT | Pares conservados por v125, sin revisión semántica | Auditar ~100 clips por programa y duración con competencia en LSA; anotar cobertura, desfase, signer y grupo | Dataset2 y metadatos restaurados | Pares completos, desfase, acuerdo entre anotadores e IC por programa | Fijar antes: ≥90% de pares completos, grupos sin leakage y regla de exclusión por desfase/desacuerdo; sin cumplirlo, no entrenar SLT | Restauración y coste experto; 100 clips no cubren todo el dominio. Justifican v118 truncado, v119 CTC y v125 integridad [E07,E08,E15] |
| 2 | PE aporta orden independientemente del resto del paquete E1 | E1 con PE, vocab121, LS0, 30 épocas y selección por edit | Apagar solo PE; mismos inits y schedule; tres seeds pareadas | Dataset1 sintético, LOSO de desarrollo 4–6 | Strict exact, edit, permutación, orden e IC por seed/signer | Mejora de edit con IC95% pareado >0 y caída consistente del orden al retirar PE | Init CIF y adaptación; prueba PE dentro de E1, no un efecto universal. Justifica el cambio confundido AR-base→E1 [E11,E12] |
| 3 | Masking estructurado reduce omisiones entre signers | E1 con STGCN congelado; E3 solo exploratorio | Keypoint masking durante entrenamiento, dosis fijada; sin descongelar simultáneamente | Dataset1 LOSO; confirmar una variante en 9–10 | Recall de glosas, strict/edit por longitud y robustez a articulaciones ausentes | Edit con IC pareado >0 y recall mejor en ambos outer; mantener orden >0,90; tres seeds de desarrollo | Artefactos de masking/gaps. Justifican fragilidad de A1 y omisiones E1 [E10,E12] |
| 4 | Contraste auxiliar activo añade grounding al AR actual | E1 CE sola; distinto del retrieval global v118 | Head auxiliar pareado, temperatura fija y negativos con B real >1; conservar decoder | Dataset1 desarrollo; dataset2 solo después de prioridad 1 | Edit/strict, shuffle, retrieval, gradiente auxiliar y variabilidad entre seeds | Gradiente no nulo y mejora pareada de edit con IC>0 en LOSO; no seleccionar solo por retrieval | Falsos negativos por glosas repetidas. Justifican NCE cero v117 y contraste global insuficiente v118 [E05–E07] |
| 5 | Curvatura supera un auxiliar euclidiano emparejado | Auxiliar euclidiano de prioridad 4 y CE sola | Cambiar únicamente proyección/métrica; curvatura fija/aprendida; igualar heads y presupuesto | Mismos datos y splits de prioridad 4; SLT certificada si existe | Edit/strict; BLEU/chrF/ROUGE-L solo en oraciones reales; rank, curvatura y gradientes | Hip>euclid con IC95% pareado >0 entre seeds/signers; no basta ganar a CE sin auxiliar | Confusión por temperatura, margen, dimensión y optimización. Justifican v118/PCA y ausencia de ensayos hiperbólicos [E07,E16,E18] |

**Experimento mínimo ahora:** prioridad 1 es una auditoría de pares, sin entrenamiento masivo. Si el objetivo inmediato es exclusivamente arquitectural dentro de E1, prioridad 2 es la ablación de un flag pendiente. No repetir CTC sobre dataset2, scheduled alpha o «InfoNCE» sin especificar target, tensor y negativos efectivos.

## 13. Evidence index

### 13.1 Fuentes y localizadores

Líneas HEAD salvo `commit:ruta`. JSON localizado por claves; directorios de eventos por versión/run. Los rangos identifican secciones inspeccionadas.

| Ref | Archivo, commit o artefacto | Evidencia |
|---|---|---|
|E01|`8dc200b:embed/train.ipynb`, celdas de código;`git log --all --reverse`;`../data/readme.MD:1–11`|prototipo Llama, HDF5 históricos|
|E02|`fccb8e1:src/mslm/training/clip_loss.py:1–31`;`ecbbd29:...:1–49`;`6350654:src/mslm/models/imitator.py`;`c200305:src/mslm/training/{softtoken_loss,padaware_loss}.py`; commits `2fb8a78`, `ce29be5`, `f13a090`, `2fa6d43`, `c200305`|contraste2025, evolución GCN/RoPE/STGCN|
|E03|`d1c95a4:src/mslm/models/imitator.py`; HEAD`src/mslm/models/imitator.py:128–306`|SignAlign/AlignmentDecoder|
|E04|`reports/v115_1.md:1–70`;`report.md:1–82`;`7b3eae0`mensaje/diff;`src/mslm/training/loss_ce_vocab.py:8–66`;`loss_msepcossim.py:4–37`;`2a70917`diff;`experiments/v115_ce_vocab/ce_vocab_v115_1.toml`|v113–115.1, pérdidas, CE floor, bug|
|E05|`eb97677:src/mslm/training/trainer.py:146–173,502–556`;`eb97677:scripts/train.py:15–51`;`src/mslm/training/loss_infonce.py:5–30`;`src/mslm/models/gemma_bridge.py:134–165,212–260`;`loss_ce_ar.py:7–62`;`imitator.py:309–341`;`be8e9f2`; configs`ce_ar_v116*.toml`,`ce_ar_v117.toml`|softprefix, frozen LM, NCE subbatch, mask/labels|
|E06|`src/mslm/training/sigreg.py:42–92`;`../outputs/reports/{113,114,115,1151,116,1161,117}/1/**/events.out.tfevents.*`; v117`16-06-2026-05-42-59/events.out.tfevents.1781588579.fischl-ubuntusv.120820.0`;`reports/cleanup_2026-06-20.md`|TB, NCE 0, chrF116.1, SIGReg, pesos eliminados|
|E07|`src/mslm/training/loss_contrastive.py:14–108`;`src/mslm/models/contrastive.py:22–89`;`scripts/train/train_contrastive_v118.py:49–77,80–125`;`experiments/v118_contrastive/*.toml`;`../outputs/reports/118/{1,2,3,4,7,8}/**/events.*`;`outputs/diag_v118_train_val_gap.json:results,gap_factor`;`report.md:236–357`;07ec4cf,daeb58b|contraste activo, proyecciones, configs y resultados|
|E08|`a48b9fe:src/mslm/models/ctc_encoder.py:1–95`;`7962692:...:1–40`; HEAD`src/mslm/models/ctc_encoder.py`;`scripts/train/train_ctc_v119.py`;`experiments/v119_ctc/*.toml`;`reports/report_2026-06-21.md`;`report.md:333–fin`;`../outputs/diagnostics/119/{1,1_baseline_no_data_levers}/`;`../outputs/reports/119/*/`|dos CTC, cantidades, posteriors/activaciones|
|E09|`reports/v120_batchnorm_regression.md`;`reports/report_2026-06-{22,23}.md`;`experiments/v121_v124_isolated_staged/*.toml`;`../outputs/experiment_summaries/v121_three_seeds.json:details,history`;`v122_ablation.json`;`src/mslm/models/isolated_classifier.py`;f088d51,35a49d5|clasificación, BN/GN, seeds, gates|
|E10|`reports/report_2026-06-24.md`,`report_2026-06-25_imitator_a1_robustness.md`,`report_2026-06-26.md`,`report_2026-06-27_etapa3_a4_scheduled_alpha.md`,`report_2026-06-28_etapa4_loso.md`,`report_2026-07-03_etapa4_loso_clean_correction.md`,`report_2026-07-03_fase0_rounded_count_prediction.md`;`../outputs/v126_temporal/*/{config.json,metrics.jsonl}`;`src/mslm/models/temporal_sign_prompt.py:22–198,200–351,375–475,509–630,694–715`; trainer`1143–1184`|CIF, oracles, stages, leakage, transferencia|
|E11|`experiments/video_token_decoder/{AUDIT_REPORT.md,HANDOFF.md,PREREG_PAPER.md,TIME_BUDGET_AMENDMENT.md,PAPER_TABLES.md}`;67641eb,3c0f703,abbed25|diagnósticos, desarrollo/confirmación, prerregistro|
|E12|`../outputs/video_token_decoder/fold{4,5,6,7,8}_seed23/e1_pe_vocab121/{config.json,metrics.jsonl,outer_test.json}`; fold 4`{base,e2_jta_ctc03,e3_unfreeze_stgcn}/`; basefold5/6;`outer_test_robustness.json`fold 7/8;`fold{4..8}_cif_comparator.json`;`src/mslm/models/video_token_decoder.py:14–66,86–145,168–239,243–284,295–354`;`scripts/train/train_video_token_decoder.py:193–228,282–317,341–358`;`src/mslm/utils/sequence_metrics.py`|arquitectura AR, free eval, resultados, linaje|
|E13|`src/mslm/dataloader/synthetic_temporal.py:32–60,125–185,189–255`;`experiments/a3_etapa4_clean_loso/manifest.json`;`scripts/loso/{build_clean_loso_manifest.py,run_clean_loso_orchestrator.py}`|generador, muestreo, gaps, LOSO|
|E14|`artifacts/data_eda/{REPORT.md,dataset_summary.csv,video_metadata.csv,eda_summary.json,metadata_unmatched.csv,srt_summary.csv}`;`docs/audits/v125_language_audit.md:1–43`;`scripts/data/build_dataset2_h5.py:1–115,147–250`;`src/mslm/dataloader/data_augmentation.py:76–96,123–150`;`../data/raw/dataset1/{meta.csv,label.csv}`|EDA, procedencia, layouts, normalización|
|E15|`../data/processed/dataset2_manifest_v125.json:summary,manifest`;`dataset2_split_v125.json:train,val,test`;`scripts/data/build_dataset2_manifest.py:31–51,54–117`;`src/mslm/dataloader/integrity_filter.py:11–54`,`grouped_split.py:13–44`;`scripts/data/backfill_dataset2_metadata.py:28–78`;`/shared/Code/Sign-AI/DATA_CLEANUP_2026-09-10.md`;`reports/cleanup_2026-06-20.md`|filtros, splits, archivos accesibles/pérdidas|
|E16|`../data/processed/v125_{artifacts,anisotropy_gate,gemma_oracle}.json`;`v125_anisotropy_gate.transform.pt`solo SHA;`scripts/audits/run_{anisotropy_gate,gemma_oracle,gemma_oracle_prompt_sweep}_v125.py`;`reports/report_2026-06-23.md`anexo;482f4e8,dee350e,d405c7d,5977e40|PCA, PLE, oracle, prompts, hashes|
|E17|`artifacts/v126_closeout/{manifest,ablation_summary}.json`;`ablation_etapa2/*_{config,gates,audit}.json`y metrics JSONL;`reports/report_2026-06-28_etapa2_ablation_causal.md`;cfdf403,7ff6015|24 runs, factorial,121 hashes|
|E18|Geo-Sign`5cacfe5b6cfc560fea5af0eb48f1544882a5844a`:README Evaluation,`models.py:56–103,158–188,235–246,282–350,375–404,455–461`,`fine_tuning.py:164–206`; búsqueda local 54 refs§6.3|geometría, mT5, loss, optimizer, reportes externos|
|E20|`new_study.db`, `study_models.db`, `../outputs/v126_temporal/optuna_v126_cif*.db`: tablas studies/trials/trial_values/trial_params, lectura SQL mode=ro; `train.log:1–60`; `scripts/train/optuna_v126_cif.py:89–128`|trials históricos y búsqueda real, sin atribuir score a accuracy|
|E19|`README.md:1–31`;`paper/content/full/sections/{03_interface_evolution,05_data_protocol,07_results}.tex`;`paper/data/frozen-paper-results/{manifest,paper_master_results}.json`;`paper/shared/references.bib:76–85`;15dc4ca,94cf338,7d20831,25576ca,a7db233|LSA64 documentado, claims, README teacher, inputs congelados|

### 13.2 Verificaciones y reproducción de lectura

Se usaron `git log --all`, `git show`, `git diff abbed25 379b942 --stat`, historial por archivo ybúsquedas por ref. Se leyeron versiones originales CTC v119, contraste2025, trainer v117 y MSE2025 para no atribuir HEAD retrospectivamente.

Inventario actual:ds1 raw3200, ds2 raw8459, outputs externos~56 MB, outputs del repo~360 KB. Cero`.pt/.pth`enoutputs externos. Siete hashes/tamañosv125 coinciden; 121entradasmanifest v126 coinciden. Split ds2 IDs disjuntos, unión=kept. Se recontó strict desde predicciones+EOS confirmatorias:112/896 y 126/896. Son comprobaciones de integridad yconsistencia, no certificación semántica.

Lectura TB sin dependencias ni escritura: parser en memoria de TFRecord(longituduint64 LE, header12 bytes, payloadEvent, CRC4); Eventsummaryfield5; SummaryValuefield1; tagfield1/simple_valuefield2 float32. Seignoraron campos desconocidos; no sevalidóCRC. v117 contiene25 NCE exactamente0; algebra y código corroboran la interpretación. Directorios con reruns mezclan archivos; máximos/counts no implican orden cronológico ni «último»porordenlexicográfico.

Ejemplo de recomprobación sin ejecutar modelos:

```python
from pathlib import Path
import json, hashlib
root = Path('/shared/Code/Sign-AI')
m = json.loads((root/'data/processed/v125_artifacts.json').read_text())
for name, record in m['artifacts'].items():
    p = root / record['path']
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    assert p.stat().st_size == record['bytes']
    assert h.hexdigest() == record['sha256'], name
```

No se ejecutópytest ni se deserializaron pesos: Torch/h5py no disponibles, binarios experimentales ausentes. Los conteos26/26,108/108, etc.de tests son DOCUMENTED históricos. No se construyóPDF por restricción de escritura; no se tomó el PDF existente como prueba depáginas.

**Estudios SQL adicionales, abiertos con `mode=ro` [E20].** `new_study.db`: 7 COMPLETE,16 FAIL,2 PRUNED; valores de objetivo almacenados entre1,48114 y 1,71236. `study_models.db`: estudios model_72/model_71,5 COMPLETE,6 FAIL,1 PRUNED,1 RUNNING histórico; objetivos1,63008–3,10511. No equivalen a procesos activos ni a métricas E1. `train.log:1–60` registra 10191 train/2547 val, encoder MLP/Conv/Transformer/proyección3072,73 M parámetros, LR7e-7 yloss1,71144 ep0; falta ligar dataset/commit ycontinuidad. No atribuir esos datos a v113 ni a ds2.

`../outputs/v126_temporal/optuna_v126_cif.db`: estudio `v126_cif_optuna_20260624`, iniciado2026-06-24 19:55:49, última finalización2026-06-25 09:22:30; 21 COMPLETE y 9 PRUNED. Maximiza score: `2*oracle_rescaled_top1 + 1.25*raw_top1 + .25*raw_top5 + .75*permutation_drop − .020*oracle_boundary_MAE − .080*raw_length_MAE − collapse_penalty`. Por incluir longitud oracle, **no es una métrica deployable ni comparable con strict exact** (`optuna_v126_cif.py:89–105`). Mejor trial 23:score 1,487591; target_only/logit_l1, lr 0,0001304967, wd 0,0000247519, alpha_weight5,22648, qty_weight1,28507, batch 4, maxclips6, gradclip1,17148, stgcn_lr_scale0,0292034. Smoke1 FAIL; smoke2 1 COMPLETE. Se decodificaron parámetros categóricos desde sus distribuciones SQL; no interpretar índices como valores.

**Tests que codifican diagnósticos:** `tests/training/test_loss_infonce.py`, `test_loss_ce_ar.py`, `test_sigreg.py`, `test_loss_ctc.py`; `tests/dataloaders/test_integrity_filter.py`, `test_grouped_split.py`; `tests/models/test_ctc_encoder.py`, `test_video_token_decoder.py`, `test_video_token_decoder_jta.py`; `tests/utils/test_sequence_metrics.py`. Documentan contratos de pérdidas, máscaras, integridad, generación y métricas. Su existencia no prueba una pasada actual ni entrenamiento.

### 13.3 Contradicciones explícitas

| Fuentes en conflicto | Resolución |
|---|---|
|resúmenesv115 cosine/10,9% vsdetallev115 dotproduct10,15%/v115.1 cosine10,87%|distinguir versiones; no reutilizar resumen como configexacta [E04]|
|v115.1«CE0,568estable»vsfix/floor|registro con escala inválida, no éxito CE [E04]|
|v117«NCEretrasócollapse9 ep»vsTB0/subbatch|NCE cero VERIFIED; atribución rechazada [E05, E06]|
|v116.1«chrF20,7 igual»vs4 registrosmáximo19,15|reportar máximo observado, no imputar intervalos [E06]|
|v113/v114«sinmétricas»vsreporte115.1/TB|hay registros, no pesos [E04, E06]|
|LOSO90,53%positivo vscorrecciónJul3|anulado por fuga upstream [E10]|
|HANDOFF/AUDIT7–10 intactosvsouter7/8|7/8 ya confirmados; 9/10 reservados soloensusrotacionesE1 [E11, E12]|
|origen ds1 UNKNOWNv125 vspaperLSA64|atribución posteriorDOCUMENTED; falta checksum deorigen [E14, E19]|
|cleanupSep10 retirólegado vsHDF5 v6 presente|v6 ysiete artefactos v125 hoypresentes/coincidentes; CSVrawausente [E15, E16]|
|CTC300 clips98,2%blank vsactualsummary5 clips99,8%|300esDOCUMENTED, noauditoríaactual; posiblesobrescritura [E08]|
|README20 queries/RoPE vsE1 CIFencoder+AR|diagramas/rutas distintas, no arquitectura exactadelcheckpointE1 [E12, E19]|

## 14. Unresolved questions

1. ¿Dónde están los backups v121, CIFclean y E1? Sin pesos no hay inicialización reutilizable inmediata ni reinferencia exacta; SHA no es backup.
2. ¿Cuál fue la orden completa y SHA de ejecución por run v113–118? Eventos/configs y commits de integración no identifican cambios no commiteados.
3. ¿Qué métricas del rerun v115.1 postfix son válidas? Múltiples eventfiles mezclan escalas; no seleccionar el número menor.
4. ¿Persisten predicciones y script de grounding v117? Retrieval0/Wilcoxon dependen de prosa; shuffle TB sí permanece.
5. ¿Hubo full runs v118d-S/T, v123/v124 en otras máquinas? Configs no permiten responder.
6. ¿Qué algoritmo, denominador y clips originaron 3088 desalineados? No equivale a 413 truncated v125.
7. ¿Cuál es la procedencia certificada de los archivos ds1/LSA64 y ds2? Faltan manifiestos de descarga y metadata cruda, sin necesidad de datos personales adicionales.
8. ¿Coincide source_group HDF5 con programa real? No se revalidó contenido interno sin h5py, aunque bytes/hashes sí coinciden.
9. ¿Qué causó CTC blank? Variación reducida del BiLSTM no separa monotonía incorrecta, desfases, escasez ni optimización.
10. ¿Cuánto aporta PE aislado? E1 funciona como paquete; falta control emparejado.
11. ¿La curvatura aporta alguna ganancia aquí? No hay ensayo local; Geo-Sign no responde por Imitator.
12. ¿Cuál es la varianza entre seeds y nuevas poblaciones de signers? Una seed y dos folds confirmatorios no la estiman.
13. ¿Puede Gemma corregir errores visuales con contexto sintáctico real? Oracle textual/glosas aisladas no lo evalúan.
14. El paper no se compiló ni se contaron páginas por autorización exclusiva del informe. Ninguna conclusión usa un PDF preexistente como evidencia actual.

**Estado defendible:** generación libre ordenada de IDs en streams sintéticos; traducción continua natural no demostrada. Antes de nuevas arquitecturas faltan supervisión certificada, pesos restaurados y controles que distingan pérdidas configuradas de pérdidas activas.
