# V126 Etapa 2 Causal Ablation + Roadmap Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unverified causal claim in Etapa 2 ("position context + attention pooling + label smoothing fixed the 3+ bottleneck") with a reproducible factorial ablation, harden the result aggregator, version closeout evidence with hashes, and make `ROADMAP_A3_CIF_LENGTH_CONDITIONED.md` internally consistent.

**Architecture:** Parametrize `TemporalSignPromptModel`'s `token_head`/`length_head` with a `{linear,contextual}` / `{mean,attention}` switch and the training script's token CE loss with a `--token-label-smoothing` flag, each independently selectable from the CLI and recorded in the checkpoint. A Python orchestrator runs the resulting 2×2×2×3-seed = 24-run matrix warm-started from the frozen Etapa 1 checkpoint, validating reuse by hashing. `analyze_imitator_a2.py` reconstructs the right architecture from each checkpoint's recorded config. A new summary tool computes mean/std/paired-seed-deltas and applies a strict causal-attribution rule. A manifest tool hashes the resulting evidence bundle. The roadmap is then edited to reflect only what the ablation actually shows.

**Tech Stack:** Python 3.13, PyTorch, pytest, existing `Sign-env` conda environment, tmux session `0` for the actual GPU runs.

## Global Constraints

- New training CLI flags, exact names/choices: `--token-head {linear,contextual}` (default `contextual`), `--length-head {mean,attention}` (default `attention`), `--token-label-smoothing FLOAT` (default `0.1`), `--split-seed INT` (default: falls back to `--seed` if omitted — preserves all existing run reproducibility).
- Checkpoints must record these params; `analyze_imitator_a2.py` must auto-reconstruct the matching architecture from a checkpoint. Checkpoints lacking this metadata default to today's behavior (`contextual`/`attention`); explicit CLI overrides on the audit script always win.
- The ablation matrix is exactly 2(token_head)×2(length_head)×2(label_smoothing)×3(seed) = 24 runs. Seeds: `23, 42, 101`. `--split-seed` is fixed at `23` for all 24 runs (split must be identical across the whole matrix; only init/sampling seed varies). 15 epochs each. Warm-start (`--resume-weights-only`) from `../outputs/v126_temporal/diag_A3_length_head_rerun_20260627_021153/checkpoint_best.pt` (the frozen Etapa 1 checkpoint).
- All 24 runs otherwise reuse the exact Etapa 2 iter2 hyperparameters (`--phase learned_cif --resume-weights-only --epochs 15 --min-clips 1 --max-clips 1 --min-neutral-frames 0 --max-neutral-frames 8 --alpha-schedule target_only --diag-alpha-loss logit_l1 --diag-freeze target_only_stage1 --stgcn-lr-scale 0.1 --prediction-alpha-mode pred_rescaled_to_pred_len`), same dataset/tokenizer/embedding table defaults, checkpoint selection by `(exact, top1)` of `pred_rescaled_to_pred_len` (already the training script's behavior — do not change it), and standalone evaluation via `analyze_imitator_a2.py`.
- A previously-run variant is only reused (skipped) if its recorded config matches exactly AND its `checkpoint_best.pt` SHA-256 matches what was recorded when the run was registered as complete. Any mismatch forces a re-run, never a silent skip.
- Ablation summary must report mean, standard deviation, and paired (same-seed) delta for: `3+` bucket `exact`, `token_accuracy_when_count_correct` (global), `count_match_rate` (global), plus the global `top1`/`top5`/`exact`/`pred_len_mae`.
- Causal attribution rule: a component (`token_head=contextual`, `length_head=attention`, or `label_smoothing=0.1`) may only be reported as a confirmed cause of an improvement in a metric if its effect is positive in all 3 seeds **both** as the marginal main effect (averaged over the other two factors) **and** when removed from the full (all-three-on) model. Otherwise the writeup must say "efecto mixto" or "interacción" — never "causa confirmada".
- The existing per-signer aggregator (`scripts/diagnostics/aggregate_loso_etapa4.py`) must reject duplicate signer entries, empty input, and incomplete sets, and must compute sample-weighted (not equal-weighted) averages. The new ablation summary tool follows the same duplicate/empty/incomplete rules for its 24 variant×seed entries.
- `artifacts/v126_closeout/` holds configs, metrics, audits, and the ablation summary — explicitly excluding model checkpoints and the redundant `predictions.jsonl`/`loso_predictions.jsonl` row dumps.
- The manifest tool needs a `build` and a `--check` subcommand. `--check` must fail (non-zero exit) on any missing file, any SHA-256 mismatch, or any incomplete/missing config field. It records: schema version, git commit hash, dataset/tokenizer identity, full config, seeds, file paths, file sizes, and SHA-256 of checkpoints and external inputs.
- Do not touch LOSO (Etapa 4), the definition of `exact`, or multi-sign evaluation — explicitly out of scope.
- Do not declare the documentation cleanup closed until all 24 runs are valid (per the reuse-hash rule) and `--check` passes on the manifest.
- Training runs are launched inside tmux session `0` (`tmux send-keys -t 0 ...`), not as ad-hoc background shells, so progress can be reattached and inspected.

---

## File Structure

- `src/mslm/models/temporal_sign_prompt.py` — add `_TokenHeadLinear`, `_LengthHeadMean`; add `token_head_variant`/`length_head_variant` params to `TemporalSignPromptModel.__init__`.
- `scripts/train/train_temporal_v126.py` — add the 4 new CLI flags; wire them into model construction, the token CE loss, the split call, and checkpoint/config persistence.
- `scripts/diagnostics/analyze_imitator_a2.py` — read `arch_config` from the checkpoint (with override flags + legacy default) to reconstruct the model; add `--split-seed`.
- `scripts/train/run_ablation_etapa2.py` (new) — resumable 24-run orchestrator with hash-based reuse validation.
- `scripts/diagnostics/aggregate_loso_etapa4.py` — harden `load_per_signer`/`average` (dedup, empty/incomplete checks, weighted average).
- `scripts/diagnostics/summarize_ablation_etapa2.py` (new) — mean/std/paired-delta + causal-attribution summary over the 24 runs' audits.
- `scripts/audits/build_v126_closeout_manifest.py` (new, modeled on `scripts/audits/freeze_v125_artifacts.py`) — `build`/`--check` manifest of `artifacts/v126_closeout/`.
- Matching tests under `tests/models/`, `tests/scripts/`.

## Interfaces Carried Across Tasks

- `TemporalSignPromptModel(frame_encoder, hidden_size, vocab_size, embedding_dim, max_len_class=16, token_head_variant="contextual", length_head_variant="attention")`. Instance exposes `.token_head_variant: str` and `.length_head_variant: str`.
- Checkpoint `state["arch_config"] = {"token_head": <str>, "length_head": <str>}`, alongside existing `epoch`/`model`/`optimizer`/`best_select_metric`/`token_ids_by_label`/`load_info`/`phase`.
- `aggregate_loso_etapa4.average(rows, weight_key="samples")` — weighted mean.
- `aggregate_loso_etapa4.validate_rows(rows, id_key="signer_id")` — raises `ValueError` on duplicate id or empty list; used by both aggregators.

---

### Task 1: Parametrize token/length heads in the model

**Files:**
- Modify: `src/mslm/models/temporal_sign_prompt.py:200-316`
- Test: `tests/models/test_v126_synthetic_temporal.py`

**Interfaces:**
- Produces: `_TokenHeadLinear(hidden_size, vocab_size, max_slots)`, `_LengthHeadMean(hidden_size, num_classes)`, both with the same `forward` signature as their `_TokenHead`/`_LengthHead` counterparts. `TemporalSignPromptModel(..., token_head_variant="contextual", length_head_variant="attention")`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/models/test_v126_synthetic_temporal.py` (near `test_token_head_with_padding_mask_produces_finite_vocab_logits` / `test_length_head_produces_batch_by_class_logits`):

```python
def test_token_head_linear_variant_ignores_cross_slot_context():
    model = TemporalSignPromptModel(
        _build_tiny_model().frame_encoder,
        hidden_size=2,
        vocab_size=10,
        embedding_dim=4,
        token_head_variant="linear",
    )
    embeddings = torch.randn(2, 4, 2)
    padding_mask = torch.tensor([[False, False, True, True], [False, True, True, True]])

    logits = model.token_head(embeddings, padding_mask)

    assert logits.shape == (2, 4, 10)
    assert torch.isfinite(logits).all()
    assert not hasattr(model.token_head, "position_embedding")


def test_length_head_mean_variant_pools_uniformly_over_valid_frames():
    model = TemporalSignPromptModel(
        _build_tiny_model().frame_encoder,
        hidden_size=2,
        vocab_size=10,
        embedding_dim=4,
        length_head_variant="mean",
    )
    features = torch.randn(3, 5, 2)
    lengths = torch.tensor([5, 3, 4])

    logits = model.length_head(features, lengths)

    assert logits.shape == (3, model.max_len_class + 1)
    assert not hasattr(model.length_head, "attn")


def test_invalid_head_variant_raises():
    with pytest.raises(ValueError):
        TemporalSignPromptModel(
            _build_tiny_model().frame_encoder,
            hidden_size=2,
            vocab_size=10,
            embedding_dim=4,
            token_head_variant="not-a-real-variant",
        )


def test_default_head_variants_match_current_behavior():
    model = TemporalSignPromptModel(
        _build_tiny_model().frame_encoder, hidden_size=2, vocab_size=10, embedding_dim=4
    )
    assert model.token_head_variant == "contextual"
    assert model.length_head_variant == "attention"
    assert hasattr(model.token_head, "position_embedding")
    assert hasattr(model.length_head, "attn")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/models/test_v126_synthetic_temporal.py -k "head_variant or invalid_head" -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'token_head_variant'`)

- [ ] **Step 3: Implement the new head classes and wire the variant switch**

In `src/mslm/models/temporal_sign_prompt.py`, immediately after the existing `_TokenHead` class (after line 234) add:

```python
class _TokenHeadLinear(nn.Module):
    """Per-slot Linear classification, no cross-slot context (pre-Etapa-2 baseline)."""

    def __init__(self, hidden_size: int, vocab_size: int, max_slots: int):
        super().__init__()
        self.max_slots = max_slots
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
```

Immediately after the existing `_LengthHead` class (after line 264) add:

```python
class _LengthHeadMean(nn.Module):
    """Mean-pooled frame summary (pre-Etapa-2 baseline)."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, frame_features: torch.Tensor, frame_lengths: torch.Tensor) -> torch.Tensor:
        mask = length_mask_from_lengths(frame_lengths, frame_features.size(1)).unsqueeze(-1)
        summed = (frame_features * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return self.classifier(summed / counts)
```

Replace `TemporalSignPromptModel.__init__` (lines 274-288) with:

```python
    def __init__(
        self,
        frame_encoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        embedding_dim: int,
        max_len_class: int = 16,
        token_head_variant: str = "contextual",
        length_head_variant: str = "attention",
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.cif = CIFAggregator(hidden_size)
        self.max_len_class = int(max_len_class)
        self.token_head_variant = token_head_variant
        self.length_head_variant = length_head_variant
        if token_head_variant == "contextual":
            self.token_head = _TokenHead(hidden_size, vocab_size, max_slots=self.max_len_class)
        elif token_head_variant == "linear":
            self.token_head = _TokenHeadLinear(hidden_size, vocab_size, max_slots=self.max_len_class)
        else:
            raise ValueError(f"unknown token_head_variant: {token_head_variant!r}")
        self.embedding_head = nn.Linear(hidden_size, embedding_dim)
        if length_head_variant == "attention":
            self.length_head = _LengthHead(hidden_size, self.max_len_class + 1)
        elif length_head_variant == "mean":
            self.length_head = _LengthHeadMean(hidden_size, self.max_len_class + 1)
        else:
            raise ValueError(f"unknown length_head_variant: {length_head_variant!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/models/test_v126_synthetic_temporal.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones — confirms no regression)

- [ ] **Step 5: Commit**

```bash
git add src/mslm/models/temporal_sign_prompt.py tests/models/test_v126_synthetic_temporal.py
git commit -m "feat: parametrize token_head/length_head variants for Etapa 2 ablation"
```

---

### Task 2: Wire CLI flags + checkpoint metadata into the training script

**Files:**
- Modify: `scripts/train/train_temporal_v126.py:56-164` (args), `:846-865` (model construction), `:790` (split call), `:961` (label smoothing), `:1112-1121` (checkpoint state)
- Test: `tests/scripts/test_train_temporal_v126_ablation_flags.py` (new)

**Interfaces:**
- Consumes: `TemporalSignPromptModel(..., token_head_variant=, length_head_variant=)` from Task 1.
- Produces: checkpoint `state["arch_config"] = {"token_head": str, "length_head": str}`; `config.json` includes `token_head`, `length_head`, `token_label_smoothing`, `split_seed` (already automatic via `vars(args)`).

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_train_temporal_v126_ablation_flags.py`:

```python
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_temporal_v126", ROOT / "scripts" / "train" / "train_temporal_v126.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["train_temporal_v126"] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _parse(extra_argv):
    sys.argv = ["train_temporal_v126.py"] + extra_argv
    return MODULE.parse_args()


def test_token_head_length_head_label_smoothing_defaults_preserve_current_behavior():
    args = _parse([])
    assert args.token_head == "contextual"
    assert args.length_head == "attention"
    assert args.token_label_smoothing == 0.1
    assert args.split_seed is None


def test_ablation_flags_are_settable():
    args = _parse(
        [
            "--token-head", "linear",
            "--length-head", "mean",
            "--token-label-smoothing", "0.0",
            "--split-seed", "23",
            "--seed", "42",
        ]
    )
    assert args.token_head == "linear"
    assert args.length_head == "mean"
    assert args.token_label_smoothing == 0.0
    assert args.split_seed == 23
    assert args.seed == 42


def test_effective_split_seed_falls_back_to_seed_when_unset():
    args = _parse(["--seed", "101"])
    assert MODULE.effective_split_seed(args) == 101


def test_effective_split_seed_uses_explicit_value():
    args = _parse(["--seed", "101", "--split-seed", "23"])
    assert MODULE.effective_split_seed(args) == 23
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_train_temporal_v126_ablation_flags.py -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'token_head'`, then later `AttributeError: module ... has no attribute 'effective_split_seed'`)

- [ ] **Step 3: Add the CLI flags**

In `scripts/train/train_temporal_v126.py`, after the `--heldout-signer` argument block (after line 82, before `--prediction-alpha-mode`) add:

```python
    parser.add_argument(
        "--token-head",
        choices=["linear", "contextual"],
        default="contextual",
        help="linear: pre-Etapa-2 per-slot Linear (no cross-slot context). "
        "contextual: position embedding + TransformerEncoderLayer (current default).",
    )
    parser.add_argument(
        "--length-head",
        choices=["mean", "attention"],
        default="attention",
        help="mean: pre-Etapa-2 mean-pooled frame summary. "
        "attention: 1-query attention pooling (current default).",
    )
    parser.add_argument(
        "--token-label-smoothing",
        type=float,
        default=0.1,
        help="label_smoothing for the token cross-entropy loss (current default preserves v126b).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for train/val split, independent of --seed (init/sampling). "
        "Defaults to --seed when omitted, preserving existing run reproducibility.",
    )
```

- [ ] **Step 4: Add `effective_split_seed` and wire it into the split call**

After `split_records` (after line 212) add:

```python
def effective_split_seed(args) -> int:
    return args.split_seed if args.split_seed is not None else args.seed
```

Replace line 790:
```python
    train_records, val_records = split_records(records, args.seed, args.heldout_signer)
```
with:
```python
    train_records, val_records = split_records(
        records, effective_split_seed(args), args.heldout_signer
    )
```

- [ ] **Step 5: Wire the head variants into model construction**

Replace the `model = TemporalSignPromptModel(...)` call (lines 846-852) with:

```python
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=args.hidden_size,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        max_len_class=args.max_len_class,
        token_head_variant=args.token_head,
        length_head_variant=args.length_head,
    ).to(device)
```

- [ ] **Step 6: Wire label smoothing into the token loss**

Replace line 961:
```python
            token_loss = F.cross_entropy(token_logits, token_targets, label_smoothing=0.1)
```
with:
```python
            token_loss = F.cross_entropy(
                token_logits, token_targets, label_smoothing=args.token_label_smoothing
            )
```

- [ ] **Step 7: Record architecture in the checkpoint**

In the `state = {...}` dict (lines 1112-1120), add an `arch_config` key:

```python
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_select_metric": best_select_metric,
            "token_ids_by_label": token_ids_by_label,
            "load_info": load_info,
            "phase": args.phase,
            "arch_config": {"token_head": args.token_head, "length_head": args.length_head},
        }
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_train_temporal_v126_ablation_flags.py tests/models/test_v126_synthetic_temporal.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/train/train_temporal_v126.py tests/scripts/test_train_temporal_v126_ablation_flags.py
git commit -m "feat: add token-head/length-head/label-smoothing/split-seed CLI flags"
```

---

### Task 3: Auto-reconstruct architecture in the audit script

**Files:**
- Modify: `scripts/diagnostics/analyze_imitator_a2.py`
- Test: `tests/scripts/test_analyze_imitator_a2_arch_reconstruction.py` (new)

**Interfaces:**
- Consumes: checkpoint `state["arch_config"]` from Task 2; `TemporalSignPromptModel(..., token_head_variant=, length_head_variant=)` from Task 1; `effective_split_seed` is re-implemented locally (this script does not import the training script's argparse-bound helper, it has its own `--seed`/`--split-seed` flags) — keep the same fallback semantics: `split_seed if split_seed is not None else seed`.
- Produces: `resolve_arch_config(checkpoint_state, cli_token_head, cli_length_head) -> dict` with keys `token_head`, `length_head`.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_analyze_imitator_a2_arch_reconstruction.py`:

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analyze_imitator_a2", ROOT / "scripts" / "diagnostics" / "analyze_imitator_a2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_arch_config_uses_checkpoint_metadata_when_present():
    state = {"arch_config": {"token_head": "linear", "length_head": "mean"}}
    resolved = MODULE.resolve_arch_config(state, cli_token_head=None, cli_length_head=None)
    assert resolved == {"token_head": "linear", "length_head": "mean"}


def test_resolve_arch_config_defaults_to_legacy_when_checkpoint_lacks_metadata():
    state = {}
    resolved = MODULE.resolve_arch_config(state, cli_token_head=None, cli_length_head=None)
    assert resolved == {"token_head": "contextual", "length_head": "attention"}


def test_resolve_arch_config_explicit_cli_override_wins():
    state = {"arch_config": {"token_head": "linear", "length_head": "mean"}}
    resolved = MODULE.resolve_arch_config(state, cli_token_head="contextual", cli_length_head=None)
    assert resolved == {"token_head": "contextual", "length_head": "mean"}


def test_effective_split_seed_falls_back_to_seed():
    assert MODULE.effective_split_seed(seed=23, split_seed=None) == 23
    assert MODULE.effective_split_seed(seed=23, split_seed=101) == 101
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_analyze_imitator_a2_arch_reconstruction.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'resolve_arch_config'`)

- [ ] **Step 3: Implement `resolve_arch_config` and `effective_split_seed`**

In `scripts/diagnostics/analyze_imitator_a2.py`, after `bucket_len` (after line 84) add:

```python
LEGACY_ARCH_CONFIG = {"token_head": "contextual", "length_head": "attention"}


def resolve_arch_config(checkpoint_state: dict, cli_token_head, cli_length_head) -> dict:
    """Pick token_head/length_head: explicit CLI > checkpoint metadata > legacy default."""
    base = dict(checkpoint_state.get("arch_config", LEGACY_ARCH_CONFIG))
    if cli_token_head is not None:
        base["token_head"] = cli_token_head
    if cli_length_head is not None:
        base["length_head"] = cli_length_head
    return base


def effective_split_seed(seed: int, split_seed) -> int:
    return split_seed if split_seed is not None else seed
```

- [ ] **Step 4: Add CLI flags and wire reconstruction into `main`**

In `parse_args` (after the `--heldout-signer` argument, after line 75) add:

```python
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--token-head", choices=["linear", "contextual"], default=None)
    parser.add_argument("--length-head", choices=["mean", "attention"], default=None)
```

In `main`, replace:
```python
    records = list_clip_records(args.h5, "dataset1")
    _, val_records = split_records(records, args.seed, args.heldout_signer)
```
with:
```python
    records = list_clip_records(args.h5, "dataset1")
    split_seed = effective_split_seed(args.seed, args.split_seed)
    _, val_records = split_records(records, split_seed, args.heldout_signer)
```

Replace the `state = torch.load(...)` / model construction block:
```python
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
```
must move to **after** `state` is loaded but model construction needs `arch_config` first, so reorder: load `state` before building `model`, then pass the resolved variants into `TemporalSignPromptModel`. Replace the existing sequence:

```python
    A = np.load("/shared/Code/Sign-AI/data/processed/adjacency_matrix.npy", allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=128)
    load_info = load_visual_low_level_weights(encoder, args.checkpoint_v121)
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=128,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=2048,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
```

with:

```python
    state = torch.load(args.checkpoint, map_location=device)
    arch_config = resolve_arch_config(state, args.token_head, args.length_head)
    A = np.load("/shared/Code/Sign-AI/data/processed/adjacency_matrix.npy", allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=128)
    load_info = load_visual_low_level_weights(encoder, args.checkpoint_v121)
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=128,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=2048,
        token_head_variant=arch_config["token_head"],
        length_head_variant=arch_config["length_head"],
    ).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
```

Add `arch_config` to the `result` dict (next to `"load_info": load_info,`):
```python
        "load_info": load_info,
        "arch_config": arch_config,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_analyze_imitator_a2_arch_reconstruction.py -v`
Expected: PASS

- [ ] **Step 6: Round-trip integration test against a real tiny checkpoint**

Add to the same test file:

```python
import torch


def test_round_trip_checkpoint_with_arch_config_loads_without_shape_mismatch(tmp_path):
    import torch.nn as nn

    from src.mslm.models.temporal_sign_prompt import TemporalSignPromptModel

    class TinyEncoder(nn.Module):
        def forward(self, keypoints, frame_lengths):
            return keypoints

    model = TemporalSignPromptModel(
        TinyEncoder(), hidden_size=4, vocab_size=12, embedding_dim=4,
        token_head_variant="linear", length_head_variant="mean",
    )
    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {"model": model.state_dict(), "arch_config": {"token_head": "linear", "length_head": "mean"}},
        ckpt_path,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    arch_config = MODULE.resolve_arch_config(state, None, None)
    rebuilt = TemporalSignPromptModel(
        TinyEncoder(), hidden_size=4, vocab_size=12, embedding_dim=4,
        token_head_variant=arch_config["token_head"], length_head_variant=arch_config["length_head"],
    )
    missing, unexpected = rebuilt.load_state_dict(state["model"], strict=True)
    assert not missing
    assert not unexpected
```

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_analyze_imitator_a2_arch_reconstruction.py -v`
Expected: PASS (strict load with zero missing/unexpected keys confirms the round-trip)

- [ ] **Step 7: Commit**

```bash
git add scripts/diagnostics/analyze_imitator_a2.py tests/scripts/test_analyze_imitator_a2_arch_reconstruction.py
git commit -m "feat: reconstruct token_head/length_head architecture from checkpoint metadata in audit script"
```

---

### Task 4: Resumable 24-run ablation orchestrator

**Files:**
- Create: `scripts/train/run_ablation_etapa2.py`
- Test: `tests/scripts/test_run_ablation_etapa2.py` (new)

**Interfaces:**
- Consumes: `--token-head`, `--length-head`, `--token-label-smoothing`, `--split-seed`, `--seed` flags from Task 2; `analyze_imitator_a2.py`'s `--split-seed`/`--token-head`/`--length-head` from Task 3.
- Produces: `generate_variants() -> list[dict]` (24 dicts, keys `token_head`, `length_head`, `token_label_smoothing`, `seed`, `split_seed`), `run_name_for(variant, run_tag) -> str`, `train_command_for(variant, run_name, resume_ckpt) -> list[str]`, `audit_command_for(variant, run_name, out_root) -> list[str]`, `is_valid_reuse(registry_entry, variant, checkpoint_path) -> bool`, `main(..., runner=subprocess.run)` (injectable for tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_run_ablation_etapa2.py`:

```python
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_ablation_etapa2", ROOT / "scripts" / "train" / "run_ablation_etapa2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_generate_variants_is_full_factorial_24_runs():
    variants = MODULE.generate_variants()
    assert len(variants) == 24
    assert len({tuple(sorted(v.items())) for v in variants}) == 24
    for v in variants:
        assert v["split_seed"] == 23
    seeds = sorted({v["seed"] for v in variants})
    assert seeds == [23, 42, 101]
    token_heads = sorted({v["token_head"] for v in variants})
    assert token_heads == ["contextual", "linear"]
    length_heads = sorted({v["length_head"] for v in variants})
    assert length_heads == ["attention", "mean"]
    smoothings = sorted({v["token_label_smoothing"] for v in variants})
    assert smoothings == [0.0, 0.1]


def test_run_name_is_deterministic_and_unique_per_variant():
    variants = MODULE.generate_variants()
    names = {MODULE.run_name_for(v, run_tag="TAG") for v in variants}
    assert len(names) == 24


def test_train_command_includes_all_ablation_flags():
    variant = {
        "token_head": "linear", "length_head": "mean",
        "token_label_smoothing": 0.0, "seed": 42, "split_seed": 23,
    }
    cmd = MODULE.train_command_for(variant, run_name="run1", resume_ckpt=Path("ckpt.pt"))
    cmd_str = " ".join(cmd)
    assert "--token-head linear" in cmd_str
    assert "--length-head mean" in cmd_str
    assert "--token-label-smoothing 0.0" in cmd_str
    assert "--seed 42" in cmd_str
    assert "--split-seed 23" in cmd_str
    assert "--resume-weights-only" in cmd_str
    assert "--epochs 15" in cmd_str


def test_is_valid_reuse_requires_matching_config_and_checkpoint_hash(tmp_path):
    ckpt = tmp_path / "checkpoint_best.pt"
    ckpt.write_bytes(b"weights-v1")
    variant = {"token_head": "linear", "length_head": "mean", "token_label_smoothing": 0.0, "seed": 23, "split_seed": 23}
    entry = {"variant": variant, "checkpoint_sha256": hashlib.sha256(b"weights-v1").hexdigest()}

    assert MODULE.is_valid_reuse(entry, variant, ckpt) is True

    ckpt.write_bytes(b"weights-v2-overwritten")
    assert MODULE.is_valid_reuse(entry, variant, ckpt) is False

    different_variant = {**variant, "seed": 999}
    ckpt.write_bytes(b"weights-v1")
    assert MODULE.is_valid_reuse(entry, different_variant, ckpt) is False


def test_is_valid_reuse_false_when_checkpoint_missing(tmp_path):
    variant = {"token_head": "linear", "length_head": "mean", "token_label_smoothing": 0.0, "seed": 23, "split_seed": 23}
    entry = {"variant": variant, "checkpoint_sha256": "deadbeef"}
    assert MODULE.is_valid_reuse(entry, variant, tmp_path / "missing.pt") is False


def test_main_skips_valid_reused_runs_and_runs_missing_ones(tmp_path):
    calls = []

    def fake_runner(cmd, **kwargs):
        # Simulate the train + audit commands writing their checkpoint.
        if "train_temporal_v126.py" in cmd[1]:
            run_name = cmd[cmd.index("--run-name") + 1]
            out_dir = tmp_path / "out" / f"diag_{run_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "checkpoint_best.pt").write_bytes(run_name.encode())
        calls.append(cmd)
        return None

    registry_dir = tmp_path / "artifacts"
    registry_dir.mkdir()
    variants = MODULE.generate_variants()[:2]  # keep the test fast
    MODULE.main(
        variants=variants,
        run_tag="TESTTAG",
        out_root=tmp_path / "out",
        registry_dir=registry_dir,
        resume_ckpt=tmp_path / "etapa1.pt",
        python_bin="python3",
        runner=fake_runner,
    )
    first_run_calls = len(calls)
    assert first_run_calls == 4  # 2 variants x (train + audit)

    # Second invocation: registries now match real checkpoints -> must skip both.
    calls.clear()
    MODULE.main(
        variants=variants,
        run_tag="TESTTAG",
        out_root=tmp_path / "out",
        registry_dir=registry_dir,
        resume_ckpt=tmp_path / "etapa1.pt",
        python_bin="python3",
        runner=fake_runner,
    )
    assert calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_run_ablation_etapa2.py -v`
Expected: FAIL (`FileNotFoundError`/`ModuleNotFoundError` — script does not exist yet)

- [ ] **Step 3: Implement `scripts/train/run_ablation_etapa2.py`**

```python
"""Resumable 2x2x2x3-seed Etapa 2 ablation orchestrator.

Run inside tmux session 0:
  tmux send-keys -t 0 \
    "PYTHONPATH=. python scripts/train/run_ablation_etapa2.py --run-tag $(date +%Y%m%d_%H%M%S)" Enter
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

TOKEN_HEADS = ("linear", "contextual")
LENGTH_HEADS = ("mean", "attention")
LABEL_SMOOTHINGS = (0.0, 0.1)
SEEDS = (23, 42, 101)
SPLIT_SEED = 23
EPOCHS = 15

DEFAULT_RESUME_CKPT = Path(
    "../outputs/v126_temporal/diag_A3_length_head_rerun_20260627_021153/checkpoint_best.pt"
)
DEFAULT_OUT_ROOT = Path("../outputs/v126_temporal")
DEFAULT_REGISTRY_DIR = Path("artifacts/v126_closeout/ablation_etapa2/registry")
DEFAULT_PYTHON_BIN = "/home/nakato/miniconda3/envs/Sign-env/bin/python"

FIXED_TRAIN_FLAGS = [
    "--phase", "learned_cif",
    "--resume-weights-only",
    "--epochs", str(EPOCHS),
    "--min-clips", "1", "--max-clips", "1",
    "--min-neutral-frames", "0", "--max-neutral-frames", "8",
    "--alpha-schedule", "target_only",
    "--diag-alpha-loss", "logit_l1",
    "--diag-freeze", "target_only_stage1",
    "--stgcn-lr-scale", "0.1",
    "--prediction-alpha-mode", "pred_rescaled_to_pred_len",
]


def generate_variants() -> list[dict]:
    variants = []
    for token_head, length_head, smoothing, seed in itertools.product(
        TOKEN_HEADS, LENGTH_HEADS, LABEL_SMOOTHINGS, SEEDS
    ):
        variants.append(
            {
                "token_head": token_head,
                "length_head": length_head,
                "token_label_smoothing": smoothing,
                "seed": seed,
                "split_seed": SPLIT_SEED,
            }
        )
    return variants


def run_name_for(variant: dict, run_tag: str) -> str:
    smoothing = variant["token_label_smoothing"]
    return (
        f"A3_etapa2_ablation_{variant['token_head']}_{variant['length_head']}"
        f"_ls{smoothing}_seed{variant['seed']}_{run_tag}"
    )


def train_command_for(variant: dict, run_name: str, resume_ckpt: Path, python_bin: str = DEFAULT_PYTHON_BIN) -> list[str]:
    return [
        python_bin, "scripts/train/train_temporal_v126.py",
        *FIXED_TRAIN_FLAGS,
        "--resume", str(resume_ckpt),
        "--token-head", variant["token_head"],
        "--length-head", variant["length_head"],
        "--token-label-smoothing", str(variant["token_label_smoothing"]),
        "--split-seed", str(variant["split_seed"]),
        "--seed", str(variant["seed"]),
        "--run-name", run_name,
    ]


def audit_command_for(variant: dict, run_name: str, out_root: Path, python_bin: str = DEFAULT_PYTHON_BIN) -> list[str]:
    checkpoint = out_root / f"diag_{run_name}" / "checkpoint_best.pt"
    output = out_root / f"diag_{run_name}_audit.json"
    return [
        python_bin, "scripts/diagnostics/analyze_imitator_a2.py",
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--split-seed", str(variant["split_seed"]),
        "--seed", str(variant["seed"]),
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_reuse(registry_entry: dict, variant: dict, checkpoint_path: Path) -> bool:
    if not checkpoint_path.is_file():
        return False
    if registry_entry.get("variant") != variant:
        return False
    return registry_entry.get("checkpoint_sha256") == sha256_of(checkpoint_path)


def main(
    *,
    variants: list[dict],
    run_tag: str,
    out_root: Path = DEFAULT_OUT_ROOT,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    resume_ckpt: Path = DEFAULT_RESUME_CKPT,
    python_bin: str = DEFAULT_PYTHON_BIN,
    runner=subprocess.run,
) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        run_name = run_name_for(variant, run_tag)
        registry_path = registry_dir / f"{run_name}.json"
        checkpoint_path = out_root / f"diag_{run_name}" / "checkpoint_best.pt"

        if registry_path.is_file():
            entry = json.loads(registry_path.read_text(encoding="utf-8"))
            if is_valid_reuse(entry, variant, checkpoint_path):
                print(f"[ablation] skip valid reuse run={run_name}")
                continue
            print(f"[ablation] registry stale/mismatched for run={run_name}, re-running")

        print(f"[ablation] run start {run_name}")
        runner(train_command_for(variant, run_name, resume_ckpt, python_bin), check=True)
        runner(audit_command_for(variant, run_name, out_root, python_bin), check=True)

        if not checkpoint_path.is_file():
            raise RuntimeError(f"training did not produce a checkpoint for {run_name}: {checkpoint_path}")
        registry_path.write_text(
            json.dumps(
                {
                    "variant": variant,
                    "run_name": run_name,
                    "checkpoint_sha256": sha256_of(checkpoint_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[ablation] run done {run_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--resume-ckpt", type=Path, default=DEFAULT_RESUME_CKPT)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        variants=generate_variants(),
        run_tag=args.run_tag,
        out_root=args.out_root,
        registry_dir=args.registry_dir,
        resume_ckpt=args.resume_ckpt,
        python_bin=args.python_bin,
    )
    print("ALL_ABLATION_RUNS_DONE", file=sys.stderr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_run_ablation_etapa2.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train/run_ablation_etapa2.py tests/scripts/test_run_ablation_etapa2.py
git commit -m "feat: add resumable 24-run Etapa 2 ablation orchestrator with hash-based reuse validation"
```

---

### Task 5: Harden the LOSO aggregator (dedup, empty/incomplete, weighted average)

**Files:**
- Modify: `scripts/diagnostics/aggregate_loso_etapa4.py`
- Modify: `tests/scripts/test_aggregate_loso_etapa4.py`

**Interfaces:**
- Produces: `validate_rows(rows, id_key="signer_id") -> None` (raises `ValueError`), `average(rows, weight_key="samples") -> dict` (sample-weighted mean, replaces the old equal-weighted version).

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_aggregate_loso_etapa4.py`:

```python
import pytest

from scripts.diagnostics.aggregate_loso_etapa4 import validate_rows


def test_average_is_weighted_by_samples():
    rows = [
        {"signer_id": 0, "samples": 10, "exact": 0.2, "top1": 0.0, "top5": 0.0,
         "pred_len_mae": 0.0, "count_match_rate": 0.0, "boundary_mae_when_count_correct": 0.0},
        {"signer_id": 1, "samples": 90, "exact": 0.8, "top1": 0.0, "top5": 0.0,
         "pred_len_mae": 0.0, "count_match_rate": 0.0, "boundary_mae_when_count_correct": 0.0},
    ]
    from scripts.diagnostics.aggregate_loso_etapa4 import average
    avg = average(rows)
    # weighted: (10*0.2 + 90*0.8) / 100 = 0.74, vs naive (0.2+0.8)/2 = 0.5
    assert avg["exact"] == pytest.approx(0.74)


def test_validate_rows_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        validate_rows([])


def test_validate_rows_rejects_duplicate_ids():
    rows = [{"signer_id": 1, "samples": 5}, {"signer_id": 1, "samples": 5}]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows)


def test_validate_rows_rejects_zero_sample_entries():
    rows = [{"signer_id": 1, "samples": 0}]
    with pytest.raises(ValueError, match="empty entry"):
        validate_rows(rows)


def test_load_per_signer_raises_on_duplicate_signer(tmp_path):
    from scripts.diagnostics.aggregate_loso_etapa4 import load_per_signer
    p1 = _write_audit(tmp_path / "s0.json", 0, **_OK)
    p2 = _write_audit(tmp_path / "s0_dup.json", 0, **_OK)
    with pytest.raises(ValueError, match="duplicate"):
        load_per_signer([p1, p2])
```

(`_write_audit`, `_OK` already exist at the top of this test file — reuse them, do not redefine.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_aggregate_loso_etapa4.py -v`
Expected: FAIL (`ImportError: cannot import name 'validate_rows'`, then weighted-average assertion failure)

- [ ] **Step 3: Implement hardening**

In `scripts/diagnostics/aggregate_loso_etapa4.py`, replace `load_per_signer` and `average` (lines 35-52):

```python
def validate_rows(rows: list[dict], id_key: str = "signer_id") -> None:
    if not rows:
        raise ValueError("aggregator received an empty row list")
    seen_ids = set()
    for row in rows:
        row_id = row[id_key]
        if row_id in seen_ids:
            raise ValueError(f"duplicate {id_key}={row_id} in aggregator input")
        seen_ids.add(row_id)
        if row.get("samples", 0) <= 0:
            raise ValueError(f"empty entry ({id_key}={row_id} has samples<=0)")


def load_per_signer(audit_paths: list[Path]) -> list[dict]:
    rows = []
    for path in audit_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        summary = data["mode_summaries"]["pred_rescaled_to_pred_len"]
        rows.append(
            {
                "signer_id": data["heldout_signer"],
                "samples": data["samples"],
                "worst_glosses": worst_glosses(data),
                **{key: summary[key] for key in METRICS},
            }
        )
    validate_rows(rows)
    return rows


def average(rows: list[dict], weight_key: str = "samples") -> dict:
    total_weight = sum(row[weight_key] for row in rows)
    return {
        key: sum(row[key] * row[weight_key] for row in rows) / total_weight
        for key in METRICS
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_aggregate_loso_etapa4.py -v`
Expected: PASS (existing tests still pass because `_OK`/`_BAD` fixtures use equal `samples=50`, so weighted == unweighted there)

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnostics/aggregate_loso_etapa4.py tests/scripts/test_aggregate_loso_etapa4.py
git commit -m "fix: weight LOSO aggregator average by samples, reject duplicate/empty entries"
```

---

### Task 6: Ablation summary tool (mean/std/paired-delta + causal attribution)

**Files:**
- Create: `scripts/diagnostics/summarize_ablation_etapa2.py`
- Test: `tests/scripts/test_summarize_ablation_etapa2.py` (new)

**Interfaces:**
- Consumes: `validate_rows` from Task 5; per-run audit JSON (`analyze_imitator_a2.py` output) plus registry JSON (`variant` dict) from Task 4.
- Produces: `load_runs(pairs) -> list[dict]` (one row per of the 24 runs: `variant` + flattened metrics), `marginal_effect(rows, factor, metric) -> dict[int, float]` (per-seed delta, averaged over the other two factors), `full_model_effect(rows, factor, metric) -> dict[int, float]` (per-seed delta when removing `factor` from the all-on combo), `classify_causal_support(marginal, full_model) -> str` (`"causa respaldada"` or `"efecto mixto"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_summarize_ablation_etapa2.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.diagnostics.summarize_ablation_etapa2 import (
    classify_causal_support,
    full_model_effect,
    load_runs,
    marginal_effect,
)

METRICS_OK = dict(
    top1=0.9, top5=0.95, exact=0.85, pred_len_mae=0.05, count_match_rate=0.9,
    token_accuracy_when_count_correct=0.95,
)


def _write_run(tmp_path, token_head, length_head, smoothing, seed, exact_3plus, **overrides):
    metrics = {**METRICS_OK, **overrides}
    variant = {
        "token_head": token_head, "length_head": length_head,
        "token_label_smoothing": smoothing, "seed": seed, "split_seed": 23,
    }
    run_name = f"run_{token_head}_{length_head}_{smoothing}_{seed}"
    registry_path = tmp_path / f"{run_name}.registry.json"
    registry_path.write_text(json.dumps({"variant": variant, "run_name": run_name}), encoding="utf-8")
    audit_path = tmp_path / f"{run_name}.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "samples": 640,
                "mode_summaries": {"pred_rescaled_to_pred_len": metrics},
                "mode_cuts": {
                    "pred_rescaled_to_pred_len": {
                        "by_target_length": {"3+": {"exact": exact_3plus, "samples": 200}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path, audit_path


def test_load_runs_rejects_duplicate_variant():
    pass  # covered indirectly via validate_rows reuse; see duplicate test below


def test_marginal_effect_is_positive_for_all_seeds_when_factor_always_helps(tmp_path):
    pairs = []
    for length_head in ("mean", "attention"):
        for smoothing in (0.0, 0.1):
            for seed in (23, 42, 101):
                # token_head=contextual always beats linear by +0.1 on 3+ exact, same seed.
                pairs.append(_write_run(tmp_path, "linear", length_head, smoothing, seed, exact_3plus=0.5))
                pairs.append(_write_run(tmp_path, "contextual", length_head, smoothing, seed, exact_3plus=0.6))
    rows = load_runs(pairs)
    effect = marginal_effect(rows, factor="token_head", metric="exact_3plus")
    assert set(effect.keys()) == {23, 42, 101}
    assert all(value == pytest.approx(0.1) for value in effect.values())


def test_classify_causal_support_requires_positive_in_all_seeds_both_ways():
    marginal = {23: 0.1, 42: 0.05, 101: 0.02}
    full_model = {23: 0.08, 42: 0.03, 101: 0.01}
    assert classify_causal_support(marginal, full_model) == "causa respaldada"


def test_classify_causal_support_is_mixed_when_one_seed_disagrees():
    marginal = {23: 0.1, 42: -0.01, 101: 0.02}
    full_model = {23: 0.08, 42: 0.03, 101: 0.01}
    assert classify_causal_support(marginal, full_model) == "efecto mixto"


def test_classify_causal_support_is_mixed_when_marginal_and_full_model_disagree():
    marginal = {23: 0.1, 42: 0.05, 101: 0.02}
    full_model = {23: -0.02, 42: 0.03, 101: 0.01}
    assert classify_causal_support(marginal, full_model) == "efecto mixto"


def test_full_model_effect_compares_all_on_combo_against_single_factor_off(tmp_path):
    pairs = [
        _write_run(tmp_path, "contextual", "attention", 0.1, 23, exact_3plus=0.86),
        _write_run(tmp_path, "linear", "attention", 0.1, 23, exact_3plus=0.61),
        _write_run(tmp_path, "contextual", "attention", 0.1, 42, exact_3plus=0.84),
        _write_run(tmp_path, "linear", "attention", 0.1, 42, exact_3plus=0.60),
    ]
    rows = load_runs(pairs)
    effect = full_model_effect(rows, factor="token_head", metric="exact_3plus")
    assert effect == {23: pytest.approx(0.25), 42: pytest.approx(0.24)}


def test_load_runs_rejects_duplicate_run(tmp_path):
    registry_path, audit_path = _write_run(tmp_path, "linear", "mean", 0.0, 23, exact_3plus=0.5)
    with pytest.raises(ValueError, match="duplicate"):
        load_runs([(registry_path, audit_path), (registry_path, audit_path)])


def test_validate_complete_rejects_anything_other_than_24_runs(tmp_path):
    from scripts.diagnostics.summarize_ablation_etapa2 import validate_complete

    pairs = [_write_run(tmp_path, "linear", "mean", 0.0, 23, exact_3plus=0.5)]
    rows = load_runs(pairs)
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete(rows)


def test_validate_complete_passes_for_full_24_run_matrix(tmp_path):
    from scripts.diagnostics.summarize_ablation_etapa2 import FACTOR_LEVELS, validate_complete

    pairs = []
    seed_counter = 0
    seeds = (23, 42, 101)
    for token_head in FACTOR_LEVELS["token_head"]:
        for length_head in FACTOR_LEVELS["length_head"]:
            for smoothing in FACTOR_LEVELS["token_label_smoothing"]:
                for seed in seeds:
                    pairs.append(_write_run(tmp_path, token_head, length_head, smoothing, seed, exact_3plus=0.5))
                    seed_counter += 1
    rows = load_runs(pairs)
    assert len(rows) == 24
    validate_complete(rows)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_summarize_ablation_etapa2.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `scripts/diagnostics/summarize_ablation_etapa2.py`**

```python
"""Mean/std/paired-seed-delta summary + causal attribution for the Etapa 2 ablation."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from scripts.diagnostics.aggregate_loso_etapa4 import validate_rows

FACTORS = ("token_head", "length_head", "token_label_smoothing")
FACTOR_LEVELS = {
    "token_head": ("linear", "contextual"),
    "length_head": ("mean", "attention"),
    "token_label_smoothing": (0.0, 0.1),
}
FULL_MODEL = {"token_head": "contextual", "length_head": "attention", "token_label_smoothing": 0.1}
GLOBAL_METRICS = ("top1", "top5", "exact", "pred_len_mae", "count_match_rate", "token_accuracy_when_count_correct")


def _run_id(variant: dict) -> tuple:
    return tuple(variant[key] for key in (*FACTORS, "seed"))


def load_runs(pairs: list[tuple[Path, Path]]) -> list[dict]:
    rows = []
    seen = set()
    for registry_path, audit_path in pairs:
        variant = json.loads(Path(registry_path).read_text(encoding="utf-8"))["variant"]
        run_id = _run_id(variant)
        if run_id in seen:
            raise ValueError(f"duplicate ablation run for variant={variant}")
        seen.add(run_id)
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        summary = audit["mode_summaries"]["pred_rescaled_to_pred_len"]
        row = {
            "variant": variant,
            "seed": variant["seed"],
            "samples": audit["samples"],
            "exact_3plus": audit["mode_cuts"]["pred_rescaled_to_pred_len"]["by_target_length"]["3+"]["exact"],
            **{key: summary[key] for key in GLOBAL_METRICS},
        }
        rows.append(row)
    validate_rows([{"signer_id": _run_id(r["variant"]), "samples": r["samples"]} for r in rows])
    return rows


def _match(row: dict, **fixed) -> bool:
    return all(row["variant"][key] == value for key, value in fixed.items())


def marginal_effect(rows: list[dict], factor: str, metric: str) -> dict[int, float]:
    """Per-seed delta for `factor`'s high level vs low level, averaged over the other two factors."""
    other_factors = [f for f in FACTORS if f != factor]
    low, high = FACTOR_LEVELS[factor]
    seeds = sorted({row["seed"] for row in rows})
    result = {}
    for seed in seeds:
        deltas = []
        for combo in itertools.product(*(FACTOR_LEVELS[f] for f in other_factors)):
            fixed_other = dict(zip(other_factors, combo))
            low_rows = [r for r in rows if r["seed"] == seed and _match(r, **{factor: low}, **fixed_other)]
            high_rows = [r for r in rows if r["seed"] == seed and _match(r, **{factor: high}, **fixed_other)]
            if not low_rows or not high_rows:
                continue
            deltas.append(high_rows[0][metric] - low_rows[0][metric])
        result[seed] = statistics.mean(deltas) if deltas else 0.0
    return result


def full_model_effect(rows: list[dict], factor: str, metric: str) -> dict[int, float]:
    """Per-seed delta between the all-on combo and the same combo with `factor` switched off."""
    low, high = FACTOR_LEVELS[factor]
    seeds = sorted({row["seed"] for row in rows})
    result = {}
    for seed in seeds:
        on_fixed = {**FULL_MODEL, factor: high}
        off_fixed = {**FULL_MODEL, factor: low}
        on_rows = [r for r in rows if r["seed"] == seed and _match(r, **on_fixed)]
        off_rows = [r for r in rows if r["seed"] == seed and _match(r, **off_fixed)]
        if on_rows and off_rows:
            result[seed] = on_rows[0][metric] - off_rows[0][metric]
    return result


def classify_causal_support(marginal: dict[int, float], full_model: dict[int, float]) -> str:
    if not marginal or not full_model:
        return "efecto mixto"
    if all(value > 0 for value in marginal.values()) and all(value > 0 for value in full_model.values()):
        return "causa respaldada"
    return "efecto mixto"


def variant_summary(rows: list[dict]) -> list[dict]:
    """Mean + stdev across seeds, grouped by the 8 non-seed variant configs."""
    by_config: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["variant"]["token_head"], row["variant"]["length_head"], row["variant"]["token_label_smoothing"])
        by_config.setdefault(key, []).append(row)
    summary_metrics = ("exact_3plus", "token_accuracy_when_count_correct", "count_match_rate", *GLOBAL_METRICS)
    out = []
    for (token_head, length_head, smoothing), group in sorted(by_config.items()):
        entry = {"token_head": token_head, "length_head": length_head, "token_label_smoothing": smoothing, "n_seeds": len(group)}
        for metric in dict.fromkeys(summary_metrics):
            values = [row[metric] for row in group]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(entry)
    return out


def validate_complete(rows: list[dict]) -> None:
    """Reject anything but the full 2x2x2x3-seed=24 matrix — partial summaries must not be reported."""
    expected = {
        (token_head, length_head, smoothing, seed)
        for token_head in FACTOR_LEVELS["token_head"]
        for length_head in FACTOR_LEVELS["length_head"]
        for smoothing in FACTOR_LEVELS["token_label_smoothing"]
        for seed in (23, 42, 101)
    }
    actual = {
        (row["variant"]["token_head"], row["variant"]["length_head"], row["variant"]["token_label_smoothing"], row["seed"])
        for row in rows
    }
    missing = expected - actual
    if missing:
        raise ValueError(f"incomplete ablation matrix: missing {len(missing)}/24 runs: {sorted(missing)}")


def causal_report(rows: list[dict]) -> dict:
    metrics = ("exact_3plus", "token_accuracy_when_count_correct", "count_match_rate", "exact")
    report = {}
    for factor in FACTORS:
        report[factor] = {}
        for metric in metrics:
            marginal = marginal_effect(rows, factor, metric)
            full_model = full_model_effect(rows, factor, metric)
            report[factor][metric] = {
                "marginal_by_seed": marginal,
                "full_model_by_seed": full_model,
                "verdict": classify_causal_support(marginal, full_model),
            }
    return report


def main(registry_dir: Path, output: Path) -> None:
    registries = sorted(registry_dir.glob("*.json"))
    pairs = []
    for registry_path in registries:
        run_name = json.loads(registry_path.read_text(encoding="utf-8"))["run_name"]
        audit_path = registry_path.parents[2] / f"diag_{run_name}_audit.json"
        pairs.append((registry_path, audit_path))
    rows = load_runs(pairs)
    validate_complete(rows)
    result = {
        "n_runs": len(rows),
        "variant_summary": variant_summary(rows),
        "causal_report": causal_report(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(output), "n_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.registry_dir, args.output)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_summarize_ablation_etapa2.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnostics/summarize_ablation_etapa2.py tests/scripts/test_summarize_ablation_etapa2.py
git commit -m "feat: add ablation summary tool with mean/std/paired-delta and strict causal attribution"
```

---

### Task 7: Closeout manifest tool (build / --check)

**Files:**
- Create: `scripts/audits/build_v126_closeout_manifest.py` (modeled on `scripts/audits/freeze_v125_artifacts.py`)
- Test: `tests/scripts/test_build_v126_closeout_manifest.py` (new)

**Interfaces:**
- Consumes: `sha256_of` pattern from `freeze_v125_artifacts.py` (reimplemented locally to keep this script standalone, same as the existing precedent does not share code across audit scripts).
- Produces: `build_manifest(root, artifact_dir, checkpoint_paths, external_inputs, commit_hash=None) -> dict` (hashes three categories: files under `artifact_dir`, the 24 ablation checkpoints — kept outside `artifact_dir` per the global constraint that excludes heavy checkpoints — and named external inputs like the dataset h5/tokenizer/embedding table), `check_manifest(manifest, root) -> list[str]` (list of problems; empty means OK), CLI `build` and `--check` subcommands.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_v126_closeout_manifest.py`:

```python
import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audits" / "build_v126_closeout_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_v126_closeout_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _make_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "v126_closeout"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "ablation_summary.json").write_text(json.dumps({"n_runs": 24}), encoding="utf-8")
    (artifact_dir / "config.json").write_text(json.dumps({"seeds": [23, 42, 101]}), encoding="utf-8")
    return artifact_dir


def _make_checkpoints_and_inputs(tmp_path):
    ckpt_dir = tmp_path / "outputs_external" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    checkpoint_paths = []
    for i in range(2):  # 2 stand-ins for the real 24-checkpoint list
        path = ckpt_dir / f"checkpoint_{i}.pt"
        path.write_bytes(f"weights-{i}".encode())
        checkpoint_paths.append(path)
    dataset_path = tmp_path / "outputs_external" / "dataset.hdf5"
    dataset_path.write_bytes(b"fake-dataset-bytes")
    external_inputs = {"dataset_h5": dataset_path}
    return checkpoint_paths, external_inputs


def test_build_manifest_hashes_artifact_files_checkpoints_and_external_inputs(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    assert manifest["schema_version"] == 1
    assert manifest["commit_hash"] == "abc123"
    paths = {entry["path"] for entry in manifest["files"]}
    assert "artifacts/v126_closeout/ablation_summary.json" in paths
    assert len(manifest["checkpoints"]) == 2
    assert manifest["external_inputs"]["dataset_h5"]["sha256"] == hashlib.sha256(
        external_inputs["dataset_h5"].read_bytes()
    ).hexdigest()
    for entry in manifest["files"] + manifest["checkpoints"]:
        full_path = tmp_path / entry["path"]
        assert entry["sha256"] == hashlib.sha256(full_path.read_bytes()).hexdigest()
        assert entry["bytes"] == full_path.stat().st_size


def test_check_manifest_passes_when_nothing_changed(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    assert MODULE.check_manifest(manifest, tmp_path) == []


def test_check_manifest_fails_on_missing_file(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    (artifact_dir / "config.json").unlink()
    problems = MODULE.check_manifest(manifest, tmp_path)
    assert any("missing" in p for p in problems)


def test_check_manifest_fails_on_missing_checkpoint(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    checkpoint_paths[0].unlink()
    problems = MODULE.check_manifest(manifest, tmp_path)
    assert any("missing" in p for p in problems)


def test_check_manifest_fails_on_hash_mismatch(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    (artifact_dir / "config.json").write_text(json.dumps({"seeds": [1]}), encoding="utf-8")
    problems = MODULE.check_manifest(manifest, tmp_path)
    assert any("hash mismatch" in p for p in problems)


def test_build_manifest_raises_when_artifact_dir_empty(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "v126_closeout"
    artifact_dir.mkdir(parents=True)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        MODULE.build_manifest(
            tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
            external_inputs=external_inputs, commit_hash="abc123",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_build_v126_closeout_manifest.py -v`
Expected: FAIL (`FileNotFoundError` on the script path / `ModuleNotFoundError`)

- [ ] **Step 3: Implement `scripts/audits/build_v126_closeout_manifest.py`**

```python
"""Build/check a SHA-256 manifest of artifacts/v126_closeout/ for the Etapa 2 ablation closeout."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _hash_entry(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def build_manifest(
    root: Path,
    artifact_dir: Path,
    checkpoint_paths: list[Path],
    external_inputs: dict[str, Path],
    commit_hash: str | None = None,
) -> dict:
    files = sorted(p for p in artifact_dir.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files found under {artifact_dir}")
    return {
        "schema_version": 1,
        "commit_hash": commit_hash if commit_hash is not None else git_commit_hash(root),
        "artifact_dir": str(artifact_dir.relative_to(root)),
        "files": [_hash_entry(path, root) for path in files],
        "checkpoints": [_hash_entry(path, root) for path in checkpoint_paths],
        "external_inputs": {
            name: _hash_entry(path, root) for name, path in external_inputs.items()
        },
    }


def _check_entries(entries: list[dict], root: Path) -> list[str]:
    problems = []
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            continue
        actual_hash = sha256_of(path)
        if actual_hash != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']} (expected {entry['sha256']}, got {actual_hash})")
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size mismatch: {entry['path']}")
    return problems


def check_manifest(manifest: dict, root: Path) -> list[str]:
    problems = _check_entries(manifest["files"], root)
    problems += _check_entries(manifest["checkpoints"], root)
    problems += _check_entries(list(manifest["external_inputs"].values()), root)
    return problems


# Checkpoints (one per ablation run) and external inputs live outside artifact_dir —
# the global constraint excludes heavy checkpoints from the versioned artifact bundle,
# but the spec still requires their hashes recorded for reproducibility.
def default_checkpoint_paths(registry_dir: Path, out_root: Path) -> list[Path]:
    paths = []
    for registry_path in sorted(registry_dir.glob("*.json")):
        run_name = json.loads(registry_path.read_text(encoding="utf-8"))["run_name"]
        paths.append(out_root / f"diag_{run_name}" / "checkpoint_best.pt")
    return paths


DEFAULT_EXTERNAL_INPUTS = {
    "dataset_h5": Path("data/processed/dataset1_isolated_v122.hdf5"),
    "embedding_table": Path("data/processed/gemma3n_embed_table.pt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/v126_closeout"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/v126_closeout/manifest.json"))
    parser.add_argument("--registry-dir", type=Path, default=Path("artifacts/v126_closeout/ablation_etapa2/registry"))
    parser.add_argument("--out-root", type=Path, default=Path("../outputs/v126_temporal"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else root / args.artifact_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest

    if args.command == "build":
        checkpoint_paths = default_checkpoint_paths(
            args.registry_dir if args.registry_dir.is_absolute() else root / args.registry_dir,
            args.out_root if args.out_root.is_absolute() else root / args.out_root,
        )
        external_inputs = {
            name: (path if path.is_absolute() else root / path)
            for name, path in DEFAULT_EXTERNAL_INPUTS.items()
        }
        manifest = build_manifest(root, artifact_dir, checkpoint_paths, external_inputs)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"wrote": str(manifest_path), "files": len(manifest["files"])}, indent=2))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = check_manifest(manifest, root)
        if problems:
            print(json.dumps({"ok": False, "problems": problems}, indent=2))
            raise SystemExit(1)
        print(json.dumps({"ok": True, "files_checked": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /shared/Code/Sign-AI/Sign-chris && /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest tests/scripts/test_build_v126_closeout_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/audits/build_v126_closeout_manifest.py tests/scripts/test_build_v126_closeout_manifest.py
git commit -m "feat: add build/check SHA-256 manifest tool for artifacts/v126_closeout/"
```

---

## Operational Execution (controller-run, not a subagent task)

These steps run actual GPU training and cannot be delegated to a code-writing subagent. The controller runs them directly after Tasks 1-7 are merged and reviewed.

1. **Launch the 24-run matrix in tmux session 0:**
   `tmux send-keys -t 0 "cd /shared/Code/Sign-AI/Sign-chris && PYTHONPATH=. /home/nakato/miniconda3/envs/Sign-env/bin/python scripts/train/run_ablation_etapa2.py --run-tag $(date +%Y%m%d_%H%M%S) 2>&1 | tee logs/ablation_etapa2_$(date +%Y%m%d_%H%M%S).log" Enter`
   Expect roughly 45 min/run x 24 ≈ 18 hours sequential (based on the Etapa 2 iter2 run's measured ~43 min wall time on the same hardware). Check progress periodically (`tmux capture-pane -t 0 -p`), do not block other work on it.
2. Once `ALL_ABLATION_RUNS_DONE` appears and all 24 registry files validate: copy the 24 `config.json`/`metrics.jsonl`/`gates.json`/audit JSON files (excluding `checkpoint_*.pt` and `predictions.jsonl`) into `artifacts/v126_closeout/ablation_etapa2/`.
3. Run `scripts/diagnostics/summarize_ablation_etapa2.py --registry-dir artifacts/v126_closeout/ablation_etapa2/registry --output artifacts/v126_closeout/ablation_summary.json`.
4. Run `scripts/audits/build_v126_closeout_manifest.py build`, then `scripts/audits/build_v126_closeout_manifest.py check` and confirm `{"ok": true, ...}`.
5. Rewrite `ROADMAP_A3_CIF_LENGTH_CONDITIONED.md`'s Etapa 2 causal conclusion using the real `causal_report` from step 3 (only — no other section may claim a cause not backed by that file). In the same edit: mark step 5 of "Próximos Pasos Inmediatos" as done (strike it like step 6 already is — it currently reads "en curso" even though Etapa 3 is closed below it), fix the Etapa 2 JSON path (`diag_A3_etapa2_audit_iter2_20260627_203817.json` → the real path, which has no `.json` extension: `diag_A3_etapa2_audit_iter2_20260627_203817`, plus its `.md` sibling), replace Etapa 3's stale absolute acceptance thresholds (`exact >= 0.818`, `top1 >= 0.869`) with a relative comparison against the current official A3 result (`exact=0.9000`, `top1=0.9165`), and move long inline historical metric dumps out to the stage report files, leaving one short pointer per stage in the roadmap.
6. Run the full suite (`pytest`), `python -c "import ast; ast.parse(open(p).read())"`-equivalent compile check is covered by pytest import already, and `git diff --check`.
7. Confirm acceptance: 24/24 runs valid per the registry, `causal_report` recomputable from the committed audits, manifest `check` passes, roadmap has no contradictory state, and the Etapa 2 conclusion matches `causal_report` exactly (no claim beyond what it shows).

## Out of Scope (explicitly deferred per the spec)

- LOSO (Etapa 4) methodology, the definition of `exact`, multi-sign evaluation — not touched by this plan.
- Re-running Etapa 3 (A4 scheduled alpha) — already closed with a negative result; not reopened here.
