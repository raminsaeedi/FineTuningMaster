"""Optional batched inference: its setting, its safety gate, its provenance.

Inference is the expensive half of the thesis matrix, and generating one item at
a time leaves most of a GPU idle. Batching several prompts into a single
``model.generate`` call is the obvious throughput win — but it is *not* a
free one, and this module exists so the trade-off is explicit rather than
implicit.

Why batching is not result-preserving
-------------------------------------
Two independent reasons, both real:

1. **Sampling.** The methods run with ``do_sample: true``. All rows of a batch
   draw from one RNG stream inside ``generate``, so the sequence of random
   numbers an item receives depends on how many items sit next to it and in
   which order. Item *k* generated in a batch of 4 is a different draw from the
   same distribution than item *k* generated alone. Same model, same prompt,
   different sample.
2. **Numerics.** Batching left-pads the prompts to a common length and changes
   the shapes fed to every matmul. Kernel selection and reduction order follow
   the shape, so even greedy decoding is only *usually* — never provably —
   identical to the sequential path.

Therefore batching is never enabled implicitly. The default is
``batch_size: 1``, which takes the untouched sequential code path, and any
larger value requires the operator to acknowledge the non-equivalence
explicitly. The resolved plan is recorded in ``manifest.json`` so a reader can
tell afterwards which regime produced a result file.

Configuration lives under the method's existing ``inference`` group (the same
group that already carries ``load_in_4bit`` for methods C and D)::

    +method.inference.batch_size=4
    +method.inference.allow_nonequivalent_batching=true

The keys are deliberately *not* written into the method YAML files: every key
present in the composed config feeds ``config_hash``, and adding a default there
would change the hash of every already-completed run, invalidating caches that
must stay readable. Absent keys resolve to the safe default here instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

BATCH_SIZE_KEY = "method.inference.batch_size"
ACKNOWLEDGE_KEY = "method.inference.allow_nonequivalent_batching"

#: ``batch_size: 1`` reproduces the sequential path exactly — same call, same
#: tensors, one item per ``generate``.
EQUIVALENCE_EXACT = "sequential-exact"
#: ``batch_size > 1`` with sampling: different RNG draw per item *and* padded
#: numerics. Not comparable item-by-item with a sequential run.
EQUIVALENCE_SAMPLING = "not-guaranteed-sampling-and-numerics"
#: ``batch_size > 1`` with greedy decoding: no RNG divergence, but padding and
#: batch-shaped kernels still make bitwise identity unproven.
EQUIVALENCE_NUMERIC = "not-guaranteed-numerics"


class UnsafeBatchingError(RuntimeError):
    """Raised when batching would silently change scientific results."""


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one key from a dict or an OmegaConf node, tolerant of missing keys."""
    if cfg is None:
        return default
    try:
        value = cfg.get(key, default)
    except AttributeError:
        value = getattr(cfg, key, default)
    return default if value is None else value


@dataclass(frozen=True)
class BatchPlan:
    """The resolved, validated batching decision for one inference run."""

    requested_batch_size: int
    batch_size: int
    do_sample: bool
    acknowledged: bool
    equivalence: str
    latency_semantics: str
    note: str

    @property
    def enabled(self) -> bool:
        return self.batch_size > 1

    def as_metadata(self) -> dict:
        """JSON-safe record for ``manifest.json``."""
        return asdict(self)


def _sequential_plan(do_sample: bool) -> BatchPlan:
    return BatchPlan(
        requested_batch_size=1,
        batch_size=1,
        do_sample=do_sample,
        acknowledged=False,
        equivalence=EQUIVALENCE_EXACT,
        latency_semantics="per-item",
        note="Default sequential inference: one item per generate call.",
    )


def resolve_batch_plan(method_cfg: Mapping[str, Any] | Any) -> BatchPlan:
    """Validate the batching settings of one method config.

    Returns the sequential plan when nothing is configured. Raises rather than
    quietly degrading whenever the requested mode cannot be honoured as asked.
    """
    inference_cfg = _get(method_cfg, "inference", {})
    generate_cfg = _get(method_cfg, "generate", {})
    do_sample = bool(_get(generate_cfg, "do_sample", True))
    constrained = bool(_get(generate_cfg, "constrained", False))

    raw = _get(inference_cfg, "batch_size", 1)
    try:
        requested = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{BATCH_SIZE_KEY} must be an integer, got {raw!r}.") from exc
    if requested < 1:
        raise ValueError(f"{BATCH_SIZE_KEY} must be >= 1, got {requested}.")
    if requested == 1:
        return _sequential_plan(do_sample)

    if constrained:
        raise UnsafeBatchingError(
            "Constrained JSON decoding generates one prompt at a time; "
            f"{BATCH_SIZE_KEY}={requested} is not supported with "
            "method.generate.constrained=true. Leave the batch size at 1."
        )

    acknowledged = bool(_get(inference_cfg, "allow_nonequivalent_batching", False))
    if not acknowledged:
        raise UnsafeBatchingError(
            f"{BATCH_SIZE_KEY}={requested} cannot be proven item-identical to "
            "sequential inference"
            + (
                " because method.generate.do_sample=true: every row of a batch "
                "draws from one shared RNG stream, so an item's sample depends "
                "on its batch neighbours"
                if do_sample
                else " because left padding and batch-shaped kernels change the "
                "numerics even under greedy decoding"
            )
            + ".\n"
            "Refusing to change results silently. Either keep the default "
            f"({BATCH_SIZE_KEY}=1, the sequential path used by every existing "
            f"result), or opt in explicitly with {ACKNOWLEDGE_KEY}=true after "
            "validating equality with:\n"
            "  python experiments/scripts/benchmark_batch_inference.py --help"
        )

    return BatchPlan(
        requested_batch_size=requested,
        batch_size=requested,
        do_sample=do_sample,
        acknowledged=True,
        equivalence=EQUIVALENCE_SAMPLING if do_sample else EQUIVALENCE_NUMERIC,
        latency_semantics="amortized-per-batch",
        note=(
            "Throughput mode, explicitly opted in. Per-item outputs are NOT "
            "comparable with sequential runs, and latency_ms is the batch wall "
            "time divided by the number of items in the batch."
        ),
    )


def batching_provenance(cfg: Any) -> dict:
    """Record the batching regime for ``manifest.json``; never raises.

    Validation failures belong to the inference run, which fails loudly on its
    own. Writing run metadata must not be the thing that explodes, so a rejected
    configuration is recorded as such instead.
    """
    method_cfg = _get(cfg, "method", {})
    try:
        return resolve_batch_plan(method_cfg).as_metadata()
    except (UnsafeBatchingError, ValueError) as exc:
        inference_cfg = _get(method_cfg, "inference", {})
        return {
            "requested_batch_size": _get(inference_cfg, "batch_size", 1),
            "batch_size": None,
            "do_sample": bool(_get(_get(method_cfg, "generate", {}), "do_sample", True)),
            "acknowledged": bool(
                _get(inference_cfg, "allow_nonequivalent_batching", False)
            ),
            "equivalence": "rejected",
            "latency_semantics": None,
            "note": f"Batching configuration rejected: {exc}",
        }
