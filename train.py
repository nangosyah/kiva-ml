"""
Train step of the Kiva ML pipeline.

Reads tidied data from the local pins board, evaluates a sector classifier
via cross-validation on the training split, fits it on the full training
set, and saves the fitted pipeline as a vetiver model pin for deployment.

The held-out test split is also pinned so a later evaluation step can score
the deployed model without ever having been used during development.
"""
import pandas as pd
import pins
import vetiver
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BOARD_PATH = "pins_board"
TIDY_PIN = "kiva_tidy_loans"
TEST_PIN = "kiva_sector_test"
MODEL_PIN = "kiva_sector_model"

RANDOM_STATE = 2984

# loan_amount was tested (see project README/blog for the ablation) and found
# to add no measurable signal beyond the description text, so the final
# pipeline uses text only.
FEATURES = ["use"]
TARGET = "sector_grouped"

SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("use_text", TfidfVectorizer(max_features=300, stop_words="english"), "use"),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def main():
    board = pins.board_folder(BOARD_PATH, allow_pickle_read=True)
    tidy = board.pin_read(TIDY_PIN)

    X = tidy[FEATURES]
    y = tidy[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline()

    results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
    print("Cross-validated performance (training split only):")
    print(f"  accuracy: {results['test_accuracy'].mean():.3f} +/- {results['test_accuracy'].std():.3f}")
    print(f"  f1_macro: {results['test_f1_macro'].mean():.3f} +/- {results['test_f1_macro'].std():.3f}")

    # Fit on the full training split for the artifact we'll deploy.
    pipe.fit(X_train, y_train)

    board.pin_write(
        pd.concat([X_test, y_test.rename(TARGET)], axis=1),
        TEST_PIN,
        type="csv",
    )

    v = vetiver.VetiverModel(
        pipe,
        model_name=MODEL_PIN,
        prototype_data=X_train,
    )
    vetiver.vetiver_pin_write(board, v)
    print(f"Saved fitted pipeline to pin '{MODEL_PIN}' and held-out test rows to '{TEST_PIN}'")


if __name__ == "__main__":
    main()
