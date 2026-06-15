#!/usr/bin/env python
import argparse
from pathlib import Path
from pprint import pprint

import pandas as pd

from modules.metrics import compute_mlc


CHEXBERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute CheXbert-label-based clinical efficacy metrics from "
            "pre-labeled generated and reference report CSV files."
        )
    )
    parser.add_argument(
        "--res_csv",
        default="results/mimic_cxr/res_labeled.csv",
        help="CSV containing CheXbert labels for generated reports.",
    )
    parser.add_argument(
        "--gts_csv",
        default="results/mimic_cxr/gts_labeled.csv",
        help="CSV containing CheXbert labels for reference reports.",
    )
    parser.add_argument(
        "--id_column",
        default=None,
        help=(
            "Optional identifier column. If omitted, the first CSV column is "
            "treated as the identifier and excluded from label columns."
        ),
    )
    parser.add_argument(
        "--label_columns",
        default=None,
        help=(
            "Optional comma-separated label columns. If omitted, the script "
            "uses the standard 14 CheXbert labels when present; otherwise it "
            "uses all columns except the identifier column."
        ),
    )
    return parser.parse_args()


def infer_columns(res_df, gts_df, id_column, label_columns):
    if id_column is None:
        id_column = res_df.columns[0]

    if label_columns:
        labels = [x.strip() for x in label_columns.split(",") if x.strip()]
    elif all(label in res_df.columns for label in CHEXBERT_LABELS) and all(
        label in gts_df.columns for label in CHEXBERT_LABELS
    ):
        labels = CHEXBERT_LABELS
    else:
        labels = [c for c in res_df.columns if c != id_column]

    missing_res = [c for c in labels if c not in res_df.columns]
    missing_gts = [c for c in labels if c not in gts_df.columns]
    if missing_res or missing_gts:
        raise ValueError(
            "Missing label columns. "
            f"res_csv missing={missing_res}; gts_csv missing={missing_gts}"
        )

    return id_column, labels


def align_by_id_if_possible(res_df, gts_df, id_column):
    if id_column in res_df.columns and id_column in gts_df.columns:
        if res_df[id_column].tolist() == gts_df[id_column].tolist():
            return res_df, gts_df

        res_indexed = res_df.set_index(id_column, drop=False)
        gts_indexed = gts_df.set_index(id_column, drop=False)
        common_ids = [idx for idx in gts_indexed.index if idx in set(res_indexed.index)]

        if not common_ids:
            raise ValueError(
                f"No overlapping identifiers found using id_column={id_column!r}."
            )

        print(
            "Warning: input CSV identifier order differs; aligning rows by "
            f"{id_column!r} over {len(common_ids)} common identifiers."
        )
        return res_indexed.loc[common_ids].reset_index(drop=True), gts_indexed.loc[
            common_ids
        ].reset_index(drop=True)

    return res_df, gts_df


def labels_to_binary(df, labels):
    values = df[labels].apply(pd.to_numeric, errors="coerce").fillna(0)
    values = values.replace(-1, 0)
    return (values == 1).astype(int).to_numpy()


def main():
    args = parse_args()

    res_path = Path(args.res_csv)
    gts_path = Path(args.gts_csv)

    if not res_path.exists():
        raise FileNotFoundError(f"Generated-report label CSV not found: {res_path}")
    if not gts_path.exists():
        raise FileNotFoundError(f"Reference-report label CSV not found: {gts_path}")

    res_df = pd.read_csv(res_path)
    gts_df = pd.read_csv(gts_path)

    id_column, label_set = infer_columns(
        res_df, gts_df, args.id_column, args.label_columns
    )
    res_df, gts_df = align_by_id_if_possible(res_df, gts_df, id_column)

    res_data = labels_to_binary(res_df, label_set)
    gts_data = labels_to_binary(gts_df, label_set)

    print(f"Generated-label CSV: {res_path}")
    print(f"Reference-label CSV: {gts_path}")
    print(f"Identifier column: {id_column}")
    print(f"Number of examples: {len(res_df)}")
    print("Label columns:")
    for label in label_set:
        print(f"  - {label}")

    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)


if __name__ == "__main__":
    main()
