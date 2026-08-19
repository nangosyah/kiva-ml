"""
Transform / tidy step of the Kiva ML pipeline.

Reads the raw extract produced by kiva_extract.py, tidies it, engineers
features, and writes the result to a local pins board so downstream steps
(train.py) can read a single source of truth.
"""
import pandas as pd
import pins

RAW_CSV = "kiva_uganda.csv"
BOARD_PATH = "pins_board"
PIN_NAME = "kiva_tidy_loans"

# Sectors with fewer than this many observations get folded into "Other"
# before applying the coarser business grouping below.
RARE_SECTOR_THRESHOLD = 20

# Final 3-class grouping used for the sector classifier.
SECTOR_GROUPS = {
    "Agriculture": "Agriculture",
    "Retail": "Business",
    "Services": "Business",
    "Housing": "Personal",
    "Construction": "Personal",
    "Food": "Personal",
    "Health": "Personal",
    "Clothing": "Personal",
    "Other": "Personal",
}


def tidy_loans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature: word count of the borrower's stated use of funds.
    df["description_length"] = df["use"].str.split().str.len()

    # Merge rare sectors, then group into the 3 coarse classes used for modeling.
    sector_counts = df["sector"].value_counts()
    rare_sectors = sector_counts[sector_counts < RARE_SECTOR_THRESHOLD].index
    df["sector_merged"] = df["sector"].where(~df["sector"].isin(rare_sectors), "Other")
    df["sector_grouped"] = df["sector_merged"].map(SECTOR_GROUPS)

    keep_cols = [
        "id",
        "status",
        "sector",
        "sector_merged",
        "sector_grouped",
        "activity",
        "loan_amount",
        "funded_amount",
        "lender_count",
        "borrower_count",
        "use",
        "description_length",
    ]
    return df[keep_cols]


def main():
    df = pd.read_csv(RAW_CSV)
    tidy = tidy_loans(df)

    board = pins.board_folder(BOARD_PATH, allow_pickle_read=True)
    board.pin_write(tidy, PIN_NAME, type="csv")
    print(f"Wrote {len(tidy)} tidied rows to pin '{PIN_NAME}' on board '{BOARD_PATH}'")
    print(tidy["sector_grouped"].value_counts())


if __name__ == "__main__":
    main()
