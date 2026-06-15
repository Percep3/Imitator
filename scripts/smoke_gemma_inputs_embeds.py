"""Smoke test: ¿inputs_embeds reproduce logits de input_ids en Gemma-3n?

Diagnóstico ONE-TIME para v116 (soft-prefix). Responde dos preguntas:

  Risk 1 — ¿inputs_embeds == input_ids logíticamente?
    Ejecuta un forward con input_ids y otro con inputs_embeds = embed_layer(ids).
    Imprime max|logits(input_ids) - logits(inputs_embeds)|.
    Umbral = 0.5 (tolerancia por cuantización 4bit).

  Risk 2 — ¿funciona forward text-only con inputs_embeds en un modelo multimodal?
    Gemma-3n se carga con AutoModelForImageTextToText.
    Se intenta primero el forward en el top-level model; si crashea, se usa el
    sub-módulo .language_model directamente.

Notas de arquitectura (verificadas al escribir este script):
  - Clase usada: AutoModelForImageTextToText (AutoModelForCausalLM falla para gemma-3n).
  - Embedding layer: model.get_input_embeddings()  →  GemmaScaledWordEmbedding
    Método alternativo: model.language_model.model.embed_tokens
  - Re-escalado interno: SÍ. GemmaScaledWordEmbedding multiplica los embeddings por
    sqrt(hidden_size) en su forward (a diferencia de Llama/Mistral donde emb.weight
    ya vive en el espacio correcto). Por eso la tabla de soft-prefix debe construirse
    llamando a emb(ids), NO leyendo emb.weight directamente.
    Ver también add_token_ids_h5.py::phase_table para la misma observación.

Uso:
    PYTHONPATH=. python scripts/smoke_gemma_inputs_embeds.py
    PYTHONPATH=. python scripts/smoke_gemma_inputs_embeds.py --model <hf-repo>
"""
import argparse
import sys

import torch

DEFAULT_MODEL = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
LOGIT_THRESHOLD = 0.5   # tolerancia para 4bit; en fp16/fp32 esperaríamos < 1e-3


# ---------------------------------------------------------------------------
# Carga del LLM  (mismo patrón que build_dataset2_h5._load_lm)
# ---------------------------------------------------------------------------

def _load_lm(llm_model: str):
    """Carga el LLM para extracción/comparación de embeddings.

    Gemma-3n es multimodal y se distribuye como bnb-4bit (unsloth); la
    quantization_config está embebida en el checkpoint, por lo que no hace
    falta pasarla explícitamente.
    """
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="cuda")
        print(f"[load] Cargado con AutoModelForCausalLM  →  {type(model).__name__}")
        return model
    except Exception as e1:
        print(f"[load] AutoModelForCausalLM no aplica ({type(e1).__name__}); probando ImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(llm_model, device_map="cuda")
        print(f"[load] Cargado con AutoModelForImageTextToText  →  {type(model).__name__}")
        return model


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _forward_with_input_ids(model, ids: torch.Tensor) -> torch.Tensor:
    """Forward estándar pasando input_ids."""
    with torch.no_grad():
        out = model(input_ids=ids)
    return out.logits  # (1, seq_len, vocab)


def _forward_with_inputs_embeds(model, embeds: torch.Tensor):
    """Forward pasando inputs_embeds.

    Intenta primero el top-level model; si lanza (porque el modelo multimodal
    requiere pixel_values o no acepta inputs_embeds directamente), lo reintenta
    en model.language_model.

    Devuelve (logits, submodule_used_str).
    """
    # Intento 1: top-level model
    try:
        with torch.no_grad():
            out = model(inputs_embeds=embeds)
        return out.logits, "top-level model"
    except Exception as e_top:
        print(f"[fwd] top-level inputs_embeds falló ({type(e_top).__name__}: {e_top})")

    # Intento 2: model.language_model (sub-módulo texto)
    if not hasattr(model, "language_model"):
        raise RuntimeError(
            "El modelo no tiene .language_model y el forward top-level falló. "
            "Revisar la arquitectura manualmente."
        )
    print("[fwd] Reintentando con model.language_model …")
    lm = model.language_model
    with torch.no_grad():
        out = lm(inputs_embeds=embeds)
    return out.logits, "model.language_model"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Smoke test inputs_embeds vs input_ids para Gemma-3n")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo del LLM")
    ap.add_argument("--seq-len", type=int, default=5, help="Longitud de la secuencia de prueba")
    ap.add_argument("--threshold", type=float, default=LOGIT_THRESHOLD,
                    help="Umbral de max|diff| para PASS (default %(default)s)")
    args = ap.parse_args()

    print("=" * 60)
    print("SMOKE TEST: inputs_embeds vs input_ids  (Gemma-3n)")
    print("=" * 60)

    # --- Carga del modelo ---
    model = _load_lm(args.model)
    model.eval()

    embed_layer = model.get_input_embeddings()
    device = embed_layer.weight.device
    hidden_size = embed_layer.weight.shape[1]
    vocab_size  = embed_layer.weight.shape[0]
    print(f"[info] Embedding layer : {type(embed_layer).__name__}")
    print(f"[info] hidden_size     : {hidden_size}")
    print(f"[info] vocab_size      : {vocab_size}")
    print(f"[info] device          : {device}")

    # --- Secuencia de prueba: IDs plausibles (BOS + algunos tokens comunes) ---
    # Usamos IDs bajos para evitar OOV; la longitud es configurable.
    ids = torch.tensor([[1] + list(range(2, 2 + args.seq_len - 1))],
                       dtype=torch.long, device=device)
    print(f"[info] Secuencia de prueba: {ids.tolist()}  (shape {tuple(ids.shape)})")

    # --- Risk 1: comparar logits input_ids vs inputs_embeds ---
    print("\n--- Risk 1: comparación de logits ---")

    embeds = embed_layer(ids)   # (1, seq_len, hidden)
    print(f"[emb]   embeds shape = {tuple(embeds.shape)},  dtype = {embeds.dtype}")

    # Risk 2 forward (inputs_embeds path) — wrapped so failures are caught cleanly
    risk2_pass = True
    submodule_used = None
    logits_emb = None
    try:
        logits_emb, submodule_used = _forward_with_inputs_embeds(model, embeds)
        print(f"[emb]   logits shape = {tuple(logits_emb.shape)}  (via {submodule_used})")
    except Exception as e_risk2:
        risk2_pass = False
        print(f"[FAIL] _forward_with_inputs_embeds lanzó: {type(e_risk2).__name__}: {e_risk2}")

    # Mirror routing: run input_ids through the SAME sub-module as inputs_embeds so
    # the Risk 1 diff is an apples-to-apples comparison.
    if risk2_pass:
        if submodule_used == "model.language_model":
            print("[ids]  Usando model.language_model(input_ids=…) para paridad de ruta")
            with torch.no_grad():
                logits_ids = model.language_model(input_ids=ids).logits
        else:
            logits_ids = _forward_with_input_ids(model, ids)
        print(f"[ids]   logits shape = {tuple(logits_ids.shape)}")

    # Alinear shapes: si se usó language_model el shape puede diferir del top-level
    # (e.g., el top-level puede devolver (1, seq, V) con V ligeramente distinto si
    # hay proyección extra). Comparamos sólo si las shapes coinciden.
    if not risk2_pass:
        # inputs_embeds forward crashed — Risk 1 is meaningless
        risk1_pass = None
        max_diff = None
    elif logits_ids.shape != logits_emb.shape:
        print(f"[WARN] shapes distintos: {logits_ids.shape} vs {logits_emb.shape}")
        print("       La comparación cuantitativa no es posible con los logits directos.")
        print("       Esto puede ocurrir si top-level y language_model tienen cabezas distintas.")
        risk1_pass = None   # indeterminado
        max_diff = None
    else:
        diff = (logits_ids.float() - logits_emb.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"max|logits(input_ids) - logits(inputs_embeds)| = {max_diff:.4f}")
        print(f"mean|diff|                                     = {mean_diff:.4f}")
        risk1_pass = max_diff < args.threshold

    # --- Risk 2: forward text-only sin pixel_values ---
    # risk2_pass ya fue fijado arriba al ejecutar _forward_with_inputs_embeds.
    print("\n--- Risk 2: forward text-only sin pixel_values ---")
    if risk2_pass:
        print(f"[OK] Forward con inputs_embeds completó usando: {submodule_used}")
    else:
        print("[FAIL] Forward con inputs_embeds no completó — ver error arriba.")
        print("RESULT: FAIL")

    # --- Resumen ---
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Submodule usado para inputs_embeds: {submodule_used}")
    if max_diff is None:
        max_diff_str = "n/a (shapes diferentes)" if risk2_pass else "n/a (forward falló)"
    else:
        max_diff_str = f"{max_diff:.4f}  (umbral = {args.threshold})"
    print(f"  max|logits diff|                  : {max_diff_str}")

    if not risk2_pass:
        r1_label = "INDETERMINATE (Risk 2 falló)"
    elif risk1_pass is None:
        r1_label = "INDETERMINATE (shapes distintos)"
    elif risk1_pass:
        r1_label = "PASS"
    else:
        r1_label = "FAIL"

    print(f"  Risk 1 (logits equivalentes)      : {r1_label}")
    print(f"  Risk 2 (text-only forward OK)     : {'PASS' if risk2_pass else 'FAIL'}")

    if risk1_pass and risk2_pass:
        print("\nRESULT: PASS")
        sys.exit(0)
    elif risk1_pass is None and risk2_pass:
        print("\nRESULT: PASS (Risk 2) / INDETERMINATE (Risk 1 — revisar shapes)")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
