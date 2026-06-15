"""GemmaBridge: frozen Gemma-3n LLM wrapper for the v116 soft-prefix pipeline.

The bridge encapsulates the LLM so that the rest of the training code never
touches the HuggingFace model directly.  It is instantiated once in
Trainer.__init__ when loss_type == "ce_ar" and stays on GPU for the entire run.

Forward path (ce_ar loss):
    keypoints → Imitator → PrefixAdapter → prefix [B, K, hidden_size]
                                              |
              concat with embed_tokens(target_ids[:, :-1])
                                              |
                                  inputs_embeds [B, K+L-1, hidden_size]
                                              |
                        GemmaBridge.forward(inputs_embeds, attn_mask)
                                              |
                                  logits [B, K+L-1, vocab_size]
"""

import torch
import torch.nn as nn


def _load_lm(model_id: str):
    """Load a causal / image-text-to-text LLM by model_id.

    Gemma-3n is multimodal, so AutoModelForCausalLM will fail with a
    ValueError/OSError; we fall back to AutoModelForImageTextToText.
    """
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    except Exception as e1:
        print(
            f"[GemmaBridge] AutoModelForCausalLM no aplica "
            f"({type(e1).__name__}); probando ImageTextToText"
        )
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, device_map="cuda")


class GemmaBridge(nn.Module):
    """Frozen Gemma-3n wrapper for soft-prefix autoregressive training (v116).

    Parameters
    ----------
    model_id:
        HuggingFace model id or local path (e.g. ``"google/gemma-3n-E2B-it"``).
    device:
        Target device string.  ``device_map="cuda"`` is passed to
        ``from_pretrained``; this parameter is kept for API symmetry.

    Attributes
    ----------
    hidden_size : int
        Embedding / hidden dimension of the LLM (e.g. 2048 for gemma-3n-E2B).
    vocab_size : int
        Size of the token vocabulary.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__()

        # Load and freeze.
        lm = _load_lm(model_id)
        lm.requires_grad_(False)
        lm.eval()
        # Store as a non-parameter attribute so that PyTorch doesn't try to
        # move it again when the bridge itself is moved (it already lives on
        # CUDA via device_map).
        self.lm = lm

        # Resolve the text sub-module for forward passes.
        # AutoModelForImageTextToText wraps a `language_model` attribute that
        # accepts inputs_embeds without requiring pixel_values.
        if hasattr(lm, "language_model"):
            self._text_model = lm.language_model
        else:
            # Plain CausalLM — the model itself accepts inputs_embeds.
            self._text_model = lm

        # Expose integer attributes for downstream modules.
        embed_layer = lm.get_input_embeddings()
        # embed_layer.weight: [vocab_size, hidden_size]
        self.vocab_size: int = embed_layer.weight.shape[0]
        self.hidden_size: int = embed_layer.weight.shape[1]
        # Cache the embedding layer to avoid re-fetching on every embed_tokens call.
        self._embed_layer = embed_layer

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids using the LLM's input embedding layer.

        The layer is ``GemmaScaledWordEmbedding``, which already applies the
        ``sqrt(hidden_size)`` scaling factor internally — do NOT scale again.

        Parameters
        ----------
        ids:
            Long tensor of shape ``[B, L]``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, L, hidden_size]``.
        """
        with torch.no_grad():
            return self._embed_layer(ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a text-only forward pass through the LLM.

        Parameters
        ----------
        inputs_embeds:
            Float tensor of shape ``[B, S, hidden_size]``.  ``S`` is the
            total sequence length (prefix tokens + text tokens - 1).
        attention_mask:
            Bool / long tensor of shape ``[B, S]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, S, vocab_size]``.
        """
        out = self._text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return out.logits
