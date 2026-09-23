from __future__ import annotations

import argparse

try:
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Full training requires joblib+pandas+scikit-learn.\n"
        "If you're on Python 3.13+ (e.g., 3.14), install Python 3.11/3.12 OR use lite mode:\n"
        "  pip install -r requirements-lite.txt\n"
        "  python -m ml.generate_synthetic_data --out data\\training_data.csv --rows 800\n"
        "  python -m ml.train_lite_model --data data\\training_data.csv --out ml\\model.json\n"
        f"\nImport error: {e}"
    )

from core.schema import MODEL_FEATURE_COLUMNS


def train(df: pd.DataFrame) -> Pipeline:
    X = df[MODEL_FEATURE_COLUMNS]
    y = df["label"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                MODEL_FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X, y)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pipe = train(df)
    joblib.dump(pipe, args.out)
    print(f"Saved model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
