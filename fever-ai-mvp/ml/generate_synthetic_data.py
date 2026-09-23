from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict

from core.schema import FeatureVector, LABELS


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _row_for_label(label: str) -> dict:
    # Synthetic and simplistic by design; educational only.
    # Values roughly resemble plausible ranges but are not medical guidance.
    if label == "viral_fever":
        fv = FeatureVector(
            temperature_f=random.uniform(99.5, 103.0),
            fever_days=random.randint(1, 4),
            platelets=random.randint(150_000, 450_000),
            wbc=random.randint(3_500, 11_000),
            crp=random.uniform(0.0, 10.0),
            headache=random.choice([0, 1]),
            rash=random.choice([0, 1]),
            chills=random.choice([0, 1]),
        )
    elif label == "dengue_like":
        fv = FeatureVector(
            temperature_f=random.uniform(100.5, 104.5),
            fever_days=random.randint(3, 10),
            platelets=random.randint(20_000, 140_000),
            wbc=random.randint(1_500, 5_000),
            crp=random.uniform(0.0, 20.0),
            headache=random.choice([0, 1]),
            rash=random.choice([0, 1]),
            chills=random.choice([0, 1]),
        )
    elif label == "bacterial":
        fv = FeatureVector(
            temperature_f=random.uniform(100.0, 104.5),
            fever_days=random.randint(2, 10),
            platelets=random.randint(150_000, 500_000),
            wbc=random.randint(9_000, 20_000),
            crp=random.uniform(10.0, 150.0),
            headache=random.choice([0, 1]),
            rash=random.choice([0, 1]),
            chills=random.choice([0, 1]),
        )
    else:  # unknown
        fv = FeatureVector(
            temperature_f=random.uniform(98.0, 105.0),
            fever_days=random.randint(0, 14),
            platelets=random.randint(50_000, 600_000),
            wbc=random.randint(1_000, 25_000),
            crp=random.uniform(0.0, 200.0),
            headache=random.choice([0, 1]),
            rash=random.choice([0, 1]),
            chills=random.choice([0, 1]),
        )

    d = asdict(fv)

    # Introduce missingness (do not impute in extraction; training can handle via imputer).
    for k in list(d.keys()):
        if random.random() < 0.08:
            d[k] = None

    d["label"] = label
    return d


def generate_rows(rows: int, *, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    per = rows // len(LABELS)
    data: list[dict] = []
    for label in LABELS:
        for _ in range(per):
            # Use module-level random for distributions, but keep class balancing stable.
            data.append(_row_for_label(label))

    while len(data) < rows:
        data.append(_row_for_label(rng.choice(LABELS)))

    rng.shuffle(data)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = generate_rows(args.rows, seed=args.seed)
    fieldnames = list(rows[0].keys()) if rows else []

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
