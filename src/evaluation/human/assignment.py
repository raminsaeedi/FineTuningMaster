"""Build the human-evaluation set and a balanced, blind rater assignment.

- ``build_eval_items`` joins each method's predictions to the gold briefs and
  selects the items that every method produced, so each item can be compared
  across all four systems.
- ``build_assignment`` creates the rater workload: every (item, method) output
  is an independent rating *unit*; each unit is rated by ``ratings_per_output``
  distinct raters, chosen to keep per-rater load balanced (a balanced incomplete
  block design). The method behind each unit is recorded for later unblinding
  but is NOT shown to raters, so rating is blind.
"""

from __future__ import annotations

import random
from typing import Dict, List


def build_eval_items(
    method_to_predictions: Dict[str, List[dict]],
    briefs_by_id: Dict[str, dict],
    n_items: int,
    seed: int = 42,
) -> List[dict]:
    """Return eval items present in every method, each with all systems' outputs.

    Each returned row:
        {"item_id", "brief", "outputs": {method: {"raw_text", "parsed"}}}
    """
    methods = list(method_to_predictions)
    by_method = {
        m: {r["item_id"]: r for r in preds}
        for m, preds in method_to_predictions.items()
    }
    common = set(briefs_by_id)
    for m in methods:
        common &= set(by_method[m])
    common_ids = sorted(common)

    rng = random.Random(seed)
    rng.shuffle(common_ids)
    chosen = common_ids[:n_items]

    items = []
    for item_id in chosen:
        outputs = {
            m: {
                "raw_text": by_method[m][item_id].get("raw_text", ""),
                "parsed": by_method[m][item_id].get("parsed"),
            }
            for m in methods
        }
        items.append({"item_id": item_id, "brief": briefs_by_id[item_id], "outputs": outputs})
    return items


def build_assignment(
    item_ids: List[str],
    methods: List[str],
    raters: List[str],
    ratings_per_output: int = 3,
    seed: int = 42,
) -> dict:
    """Assign each (item, method) unit to ``ratings_per_output`` distinct raters.

    Returns a dict with per-rater task lists (in randomized order) plus config.
    Raises if ``ratings_per_output`` exceeds the number of raters.
    """
    if ratings_per_output > len(raters):
        raise ValueError(
            f"ratings_per_output ({ratings_per_output}) cannot exceed number of "
            f"raters ({len(raters)})."
        )

    rng = random.Random(seed)
    units = [(item_id, m) for item_id in item_ids for m in methods]
    rng.shuffle(units)

    load = {r: 0 for r in raters}
    per_rater: Dict[str, List[dict]] = {r: [] for r in raters}

    for item_id, method in units:
        # Choose the k least-loaded raters (ties broken randomly) for this unit.
        ordered = sorted(raters, key=lambda r: (load[r], rng.random()))
        chosen = ordered[:ratings_per_output]
        unit_id = f"{item_id}__{method}"
        for r in chosen:
            per_rater[r].append({"unit_id": unit_id, "item_id": item_id, "method": method})
            load[r] += 1

    # Randomize each rater's task order so method order is not predictable.
    for r in raters:
        rng.shuffle(per_rater[r])

    return {
        "config": {
            "methods": methods,
            "raters": raters,
            "ratings_per_output": ratings_per_output,
            "n_items": len(item_ids),
            "n_units": len(units),
            "seed": seed,
        },
        "load": load,
        "raters": per_rater,
    }
