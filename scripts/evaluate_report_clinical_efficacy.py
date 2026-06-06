"""
Report-level clinical efficacy evaluation for SEDRRG.

This script computes micro-averaged precision, recall, and F1 over the
14-label CheXpert-style observation set after generated and reference reports
have been converted to binary clinical labels by the same report-labeling
protocol. It does not distribute or invoke any restricted clinical report
labeler; users should provide the label outputs produced under their local
data-use conditions.

Expected input formats
----------------------
1) Paired JSON/JSONL file via --input_path, where each item contains:
   - id
   - generated_ce_labels or pred_labels or clinical_labels
   - reference_ce_labels or ref_labels or ce_labels

2) Separate generated and reference JSON/JSONL files via --generated_path and
   --reference_path. Items are matched by id. Generated items should contain
   generated_ce_labels / pred_labels / clinical_labels. Reference items should
   contain reference_ce_labels / ref_labels / ce_labels.

Each label vector must contain 14 binary values in the label order defined in
CLINICAL_LABELS below.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


CLINICAL_LABELS = [
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

PRED_KEYS = ["generated_ce_labels", "pred_labels", "clinical_labels", "ce_pred_labels", "ce_labels_pred"]
REF_KEYS = ["reference_ce_labels", "ref_labels", "ce_labels", "clinical_reference_labels", "reference_labels"]


def load_json_or_jsonl(path: str) -> List[dict]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)

    text = path_obj.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path_obj.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["items", "samples", "results", "outputs", "data"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        if all(split in obj for split in ["train", "val", "test"]):
            rows = []
            for split in ["train", "val", "test"]:
                rows.extend(obj.get(split, []))
            return rows
    raise ValueError(f"Unsupported JSON structure in {path}")


def pick_vector(item: dict, keys: Iterable[str], item_id: str, role: str) -> List[int]:
    for key in keys:
        if key in item:
            values = item[key]
            if len(values) != len(CLINICAL_LABELS):
                raise ValueError(
                    f"{role} label vector for id={item_id} has length {len(values)}, "
                    f"expected {len(CLINICAL_LABELS)}"
                )
            return [int(v) for v in values]
    raise KeyError(f"Cannot find {role} label vector for id={item_id}. Tried keys: {list(keys)}")


def compute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = (y_pred * y_true).sum(axis=0)
    fp = (y_pred * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_pred) * y_true).sum(axis=0)

    precision_c = tp / np.maximum(tp + fp, 1)
    recall_c = tp / np.maximum(tp + fn, 1)
    f1_c = 2 * precision_c * recall_c / np.maximum(precision_c + recall_c, 1e-8)

    micro_tp = int(tp.sum())
    micro_fp = int(fp.sum())
    micro_fn = int(fn.sum())

    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-8)

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(f1_c)),
        "counts": {
            "micro_tp": micro_tp,
            "micro_fp": micro_fp,
            "micro_fn": micro_fn,
            "num_samples": int(y_true.shape[0]),
            "num_labels": int(y_true.shape[1]),
        },
        "per_label": {
            name: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "tp": int(t),
                "fp": int(fp_i),
                "fn": int(fn_i),
            }
            for name, p, r, f, t, fp_i, fn_i in zip(
                CLINICAL_LABELS, precision_c, recall_c, f1_c, tp, fp, fn
            )
        },
    }


def build_pairs_from_paired_file(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = load_json_or_jsonl(path)
    y_pred, y_true, ids = [], [], []
    for index, item in enumerate(rows):
        item_id = str(item.get("id", index))
        ids.append(item_id)
        y_pred.append(pick_vector(item, PRED_KEYS, item_id, "generated"))
        y_true.append(pick_vector(item, REF_KEYS, item_id, "reference"))
    return np.asarray(y_true, dtype=np.int32), np.asarray(y_pred, dtype=np.int32), ids


def build_pairs_from_separate_files(generated_path: str, reference_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    generated_rows = load_json_or_jsonl(generated_path)
    reference_rows = load_json_or_jsonl(reference_path)

    ref_by_id = {str(item.get("id")): item for item in reference_rows}
    y_pred, y_true, ids = [], [], []

    for index, gen_item in enumerate(generated_rows):
        item_id = str(gen_item.get("id", index))
        if item_id not in ref_by_id:
            raise KeyError(f"Generated item id={item_id} not found in reference file")
        ref_item = ref_by_id[item_id]
        ids.append(item_id)
        y_pred.append(pick_vector(gen_item, PRED_KEYS, item_id, "generated"))
        y_true.append(pick_vector(ref_item, REF_KEYS, item_id, "reference"))

    return np.asarray(y_true, dtype=np.int32), np.asarray(y_pred, dtype=np.int32), ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate report-level clinical efficacy labels.")
    parser.add_argument("--input_path", default=None, help="Paired JSON/JSONL file with generated and reference CE labels.")
    parser.add_argument("--generated_path", default=None, help="Generated-report label JSON/JSONL file.")
    parser.add_argument("--reference_path", default=None, help="Reference-report label JSON/JSONL file.")
    parser.add_argument("--output_path", required=True, help="Path to save the metric JSON file.")
    args = parser.parse_args()

    if args.input_path:
        y_true, y_pred, ids = build_pairs_from_paired_file(args.input_path)
    else:
        if not args.generated_path or not args.reference_path:
            raise ValueError("Provide either --input_path or both --generated_path and --reference_path.")
        y_true, y_pred, ids = build_pairs_from_separate_files(args.generated_path, args.reference_path)

    if y_true.size == 0:
        raise ValueError("No evaluation samples were loaded.")

    scores = compute_scores(y_true, y_pred)
    result = {
        "label_names": CLINICAL_LABELS,
        "metrics": scores,
        "evaluated_ids": ids,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": scores}, indent=2))
    print(f"Saved report-level clinical efficacy metrics to: {output_path}")


if __name__ == "__main__":
    main()
