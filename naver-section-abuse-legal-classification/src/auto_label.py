"""
Heuristic auto labeling pipeline.
"""
import argparse
import glob
import os
from typing import List

import pandas as pd

from src.data_loader import load_raw_comments
from src.heuristic_labeler import label_comment
from src.labeling_rules import get_label_name


def load_raw_data(input_paths: List[str]) -> pd.DataFrame:
    if not input_paths:
        return load_raw_comments()

    frames = []
    for path in input_paths:
        if os.path.isdir(path):
            csv_paths = glob.glob(os.path.join(path, "*.csv"))
        else:
            csv_paths = [path]
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError("No input CSV files found.")

    return pd.concat(frames, ignore_index=True)


def auto_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    for _, row in df.iterrows():
        comment = str(row["comment_text"])
        label_info = label_comment(comment)
        predictions.append(label_info)
    pred_df = pd.DataFrame(predictions)
    return pd.concat([df, pred_df], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Heuristic auto-labeling for abusive comments.")
    parser.add_argument(
        "--input",
        nargs="*",
        default=["data/raw"],
        help="Input CSV file(s) or directory containing comments_* files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/comments_labeled_auto.csv",
        help="Output path for auto-labeled CSV.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence to accept the heuristic label.",
    )
    args = parser.parse_args()

    df = load_raw_data(args.input)
    print(f"Loaded {len(df)} comments for auto-labeling.")

    labeled_df = auto_label_dataframe(df)
    labeled_df["accepted_label"] = labeled_df.apply(
        lambda row: row["predicted_label"] if row["confidence"] >= args.min_confidence else "F",
        axis=1,
    )
    labeled_df["accepted_label_name"] = labeled_df["accepted_label"].apply(get_label_name)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    labeled_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved auto-labeled data to {args.output}")
    print("Label distribution (accepted labels):")
    print(labeled_df["accepted_label"].value_counts())


if __name__ == "__main__":
    main()

