"""Build unified DROID_val.json for cleaner eval AP.

Collapses free-form variants of compound nouns to their head noun:
  yellow block / green block / wooden block / lego block ... -> block
  glass lid / pot lid / silver lid / clear lid ...           -> lid
  spice bottle / water bottle / spray bottle ...             -> bottle
  rubik's cube / wooden cube / yellow cube ...               -> cube
  paper towel / kitchen towel / white towel ...              -> towel
  ...

Rule:
  - Targets = single-word categories with >= FREQ_THRESHOLD occurrences in
    train+val (45 cats at threshold=50).
  - Each multi-word cat is mapped iff its LAST token is a target.
  - Otherwise the cat is kept as-is (no fuzzy / substring matching to
    avoid bugs like 'orange ring' -> 'orange').

Outputs (under <droid_dir>/annotations):
  DROID_val_unified.json     - eval annotation
  DROID_unification_map.json - {raw: unified} dict

Run:
  python scripts/data_prep/droid/build_unified_val.py \\
      --droid_dir data/droid \\
      --freq_threshold 50
"""
import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--droid_dir", default="data/droid",
        help="Root containing annotations/DROID_{train,val}.json.",
    )
    parser.add_argument(
        "--freq_threshold", type=int, default=50,
        help="Minimum train+val occurrences for a single-word "
        "category to be eligible as a unification target.",
    )
    args = parser.parse_args()

    ann_dir = os.path.join(args.droid_dir, "annotations")
    src = os.path.join(ann_dir, "DROID_val.json")
    dst = os.path.join(ann_dir, "DROID_val_unified.json")
    map_path = os.path.join(ann_dir, "DROID_unification_map.json")

    all_cnt = Counter()
    for split in ("train", "val"):
        with open(os.path.join(ann_dir, f"DROID_{split}.json")) as f:
            d = json.load(f)
        for ann in d["annotations"]:
            all_cnt[ann["category_name"]] += 1

    frequent_targets = sorted(
        [c for c, n in all_cnt.items()
         if n >= args.freq_threshold and " " not in c],
        key=lambda x: -all_cnt[x],
    )
    print(f"Single-word frequent targets (>={args.freq_threshold}): "
          f"{len(frequent_targets)}")
    target_set = set(frequent_targets)

    def unify(name):
        tokens = name.split()
        last = tokens[-1] if tokens else name
        return last if last in target_set else name

    mapping = {c: unify(c) for c in all_cnt}
    n_mapped = sum(1 for c, u in mapping.items() if u != c)
    unique_after = len(set(mapping.values()))
    print(f"Mapped {n_mapped}/{len(mapping)} cats. "
          f"Unique after: {unique_after}")

    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)

    with open(src) as f:
        val = json.load(f)
    unified_set = sorted({mapping[c] for c in mapping})
    cat_to_id = {n: i for i, n in enumerate(unified_set)}
    val["categories"] = [
        {"id": i, "name": n, "supercategory": "object"}
        for n, i in cat_to_id.items()
    ]
    for ann in val["annotations"]:
        u = mapping[ann["category_name"]]
        ann["category_name"] = u
        ann["category_id"] = cat_to_id[u]
    with open(dst, "w") as f:
        json.dump(val, f)

    val_cnt = Counter(a["category_name"] for a in val["annotations"])
    print(f"\nVal: {len(val_cnt)} unique cats present (was 210)")
    print("Top 15:")
    for c, n in val_cnt.most_common(15):
        print(f"  {n:5d}  {c}")
    print(f"\nWrote {dst}")
    print(f"Wrote {map_path}")


if __name__ == "__main__":
    main()
