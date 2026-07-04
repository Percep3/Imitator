# Mapa de trazabilidad bibliográfica

Regla del manuscrito: `knowledge/` es el corpus local canónico. Toda afirmación de
related work, comparación conceptual o limitación respaldable por ese corpus debe citar
el paper local correspondiente. Se admiten fuentes externas cuando aportan la fuente
primaria o cubren un hueco, tras verificación web.

## Cobertura de `knowledge/markdown`

| Archivo local | Clave BibTeX | Uso en el borrador |
|---|---|---|
| `41598_2025_Article_32768.md` | `harrouch2026manual` | manual/non-manual streams; límites de isolated→continuous; pose estimation |
| `Contextual_Embeddings_Encode_Word_Knowledge_Anisotropy.md` | `ethayarajh2019contextual` | anisotropía y cautela al interpretar geometría de embeddings |
| `Continuous_Sign_Language_Recognition_with_Multi-Sc.md` | `wang2025stnet` | información temporal fina y modelado multi-escala en CSLR |
| `Deep_Learning_Based_Sign_Language_Recognition_Usin.md` | `yenisari2025multifeature` | atención y fusión de múltiples features |
| `FLa_LLM_Factorized_Learning_Assisted_LLM_for_Sign_Language_Translation.md` | `chen2024flallm` | separación del aprendizaje visual y el LLM |
| `Joe_Huamani.md` | `huamaniLessIsMore` | overfitting de Transformers en datasets pequeños |
| `LREC26-26014.md` | `zhao2026multimodal` | handshape, señales multimodales y detección de boundaries |
| `LegoSLM_Language_Modeling_with_Embedded_Speech_Units.md` | `ma2025legoslm` | CTC posteriors y conexión encoder→espacio de tokens LLM |
| `Min_A_Closer_Look_at_Skeleton-based_Continuous_Sign_Language_Recognition_ICCVW_2025_paper.md` | `min2025skeleton` | skeleton-CSLR, CTC y generalización limitada por datos |
| `Nam_LiftSign_at_SignEval_2026_A_Distilled_Multi-Stream_Ensemble_with_Temporal_CVPRW_2026_paper.md` | `nam2026liftsign` | skeleton multi-stream, preservación temporal y signer generalization |
| `SM4465.md` | `pai2026tsl` | pipeline ST-GCN→generación lingüística en bajo recurso |
| `Sign2GPT_Leveraging_Large_Language_Models_for_Gloss_Free_Sign_Language_Translation.md` | `wong2024sign2gpt` | interfaz visual→LLM gloss-free |
| `SignLLM_LLMs_are_Good_Sign_Language_Translators.md` | `gong2024signllm` | tokens visuales discretos y jerárquicos para LLM |
| `Text_CTC_Alignment_for_Continuous_Sign_Language_Translation.md` | `tan2025textctc` | joint CTC/attention y alineación no monotónica en SLT |
| `Wav2Prompt_Bridging_Speech_and_Text_LLMs.md` | `deng2025wav2prompt` | prompts continuos y CIF para conectar modalidades |

`SM4465.md` no aparece actualmente en `knowledge/kb/papers_metadata.json`, aunque sí
está convertido y marcado como extraído en `pipeline_status.json`; por eso su entrada se
verificó directamente en el markdown y se añadió manualmente a `references.bib`.

## Fuentes primarias externas verificadas

| Clave BibTeX | Motivo | Fuente verificada |
|---|---|---|
| `vaswani2017attention` | Transformer y positional encoding | NeurIPS proceedings |
| `yan2018stgcn` | definición original de ST-GCN | AAAI proceedings, DOI `10.1609/aaai.v32i1.12328` |
| `graves2006ctc` | definición original de CTC | ACM ICML, DOI `10.1145/1143844.1143891` |
| `camgoz2020signtransformers` | Transformer conjunto CSLR/SLT | CVF Open Access |
| `efron1979bootstrap` | bootstrap no paramétrico | Annals of Statistics, DOI `10.1214/aos/1176344552` |

Antes de añadir nuevas citas externas se debe: buscar la fuente primaria, verificar
autores/título/venue/DOI y registrar aquí qué afirmación sustenta.
