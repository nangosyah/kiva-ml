"""
Final evaluation step: score the deployed model pin on the held-out test
split (pinned by train.py). Run this exactly once, after model development
is complete.
"""
import pandas as pd
import pins
import vetiver
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BOARD_PATH = "pins_board"
MODEL_PIN = "kiva_sector_model"
TEST_PIN = "kiva_sector_test"
TARGET = "sector_grouped"


def main():
    board = pins.board_folder(BOARD_PATH, allow_pickle_read=True)
    v = vetiver.VetiverModel.from_pin(board, MODEL_PIN)
    test = board.pin_read(TEST_PIN)

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    y_pred = v.model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test f1_macro: {f1_macro:.3f}")
    print()
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))


if __name__ == "__main__":
    main()
