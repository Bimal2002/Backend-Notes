from __future__ import annotations

import argparse
from ml.lite_model import save_model, train_from_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = train_from_csv(args.data)
    save_model(model, args.out)
    print(f"Saved lite model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
