import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from models.r2gen import R2GenModel


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


def compute_scores(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = (y_pred * y_true).sum(axis=0)
    fp = (y_pred * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_pred) * y_true).sum(axis=0)

    precision_c = tp / np.maximum(tp + fp, 1)
    recall_c = tp / np.maximum(tp + fn, 1)
    f1_c = 2 * precision_c * recall_c / np.maximum(precision_c + recall_c, 1e-8)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()

    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-8)

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(f1_c)),
        "per_label_f1": {
            name: float(value) for name, value in zip(CLINICAL_LABELS, f1_c)
        },
    }


def load_annotations(path: str):
    with open(path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    id_to_labels = {}
    for split in ["train", "val", "test"]:
        if split not in annotations:
            continue
        for item in annotations[split]:
            if "ce_labels" not in item:
                raise KeyError(f"Missing ce_labels for sample {item.get('id')}")
            id_to_labels[str(item["id"])] = torch.tensor(item["ce_labels"], dtype=torch.float32)

    return annotations, id_to_labels


def labels_for_ids(ids, id_to_labels, device):
    return torch.stack([id_to_labels[str(i)] for i in ids], dim=0).to(device)


def pooled_visual_feature(visual_extractor, images):
    att_feats, fc_feats = visual_extractor(images)
    features = []

    for tensor in [fc_feats, att_feats]:
        if tensor is None:
            continue
        if tensor.dim() == 2:
            features.append(tensor)
        elif tensor.dim() == 3:
            features.append(tensor.mean(dim=1))
        elif tensor.dim() > 3:
            features.append(tensor.flatten(2).mean(dim=-1))

    if not features:
        raise RuntimeError("The visual extractor did not return usable features.")

    return torch.cat(features, dim=-1)


def build_model_args(args):
    return Namespace(
        image_dir=args.image_dir,
        ann_path=args.ann_path,
        dataset_name=args.dataset_name,
        max_seq_length=args.max_seq_length,
        threshold=args.vocab_threshold,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        visual_extractor=args.visual_extractor,
        visual_extractor_pretrained=args.visual_extractor_pretrained,
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_vf=args.d_vf,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        logit_layers=1,
        bos_idx=0,
        eos_idx=0,
        pad_idx=0,
        use_bn=0,
        drop_prob_lm=0.0,
        rm_num_slots=3,
        rm_num_heads=8,
        rm_d_model=args.d_model,
        sample_method="beam_search",
        beam_size=2,
        temperature=1.0,
        sample_n=1,
        group_size=1,
        output_logsoftmax=1,
        decoding_constraint=0,
        block_trigrams=1,
        n_gpu=1,
        num_diffusion_steps=args.num_diffusion_steps,
        sample_max_len=args.sample_max_len,
        sample_temperature=0.6,
        sample_top_k=4,
        sample_ngram_boost=0.0,
        sample_min_len=5,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIMIC-CXR clinical efficacy labels.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])

    parser.add_argument("--dataset_name", default="mimic_cxr", choices=["iu_xray", "mimic_cxr"])
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--vocab_threshold", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--visual_extractor", default="swim_transformer_v2")
    parser.add_argument("--visual_extractor_pretrained", action="store_true")
    parser.add_argument("--feature_dim", type=int, default=768)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--d_vf", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=9)
    parser.add_argument("--sample_max_len", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    _, id_to_labels = load_annotations(args.ann_path)
    model_args = build_model_args(args)

    tokenizer = Tokenizer(model_args)
    model_args.vocab_size = tokenizer.get_vocab_size()
    model_args.pad_idx = int(tokenizer.pad_token_id)
    model_args.eos_idx = int(tokenizer.eos_token_id)
    model_args.bos_idx = int(tokenizer.cls_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = R2DataLoader(model_args, tokenizer, split=args.split, shuffle=False)
    base_model = R2GenModel(model_args, tokenizer).to(device)
    visual_extractor = base_model.visual_extractor

    ids, images, _, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        feature_dim = pooled_visual_feature(visual_extractor, images).size(-1)

    classifier = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, len(CLINICAL_LABELS)),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    visual_extractor.load_state_dict(checkpoint["visual"], strict=False)
    classifier.load_state_dict(checkpoint["head"], strict=False)

    threshold = float(args.threshold) if args.threshold is not None else float(checkpoint.get("threshold", 0.5))

    visual_extractor.eval()
    classifier.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for ids, images, _, _ in loader:
            images = images.to(device)
            labels = labels_for_ids(ids, id_to_labels, device)
            features = pooled_visual_feature(visual_extractor, images)
            logits = classifier(features)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    scores = compute_scores(y_true, y_prob, threshold)

    result = {
        "split": args.split,
        "threshold": threshold,
        "label_names": CLINICAL_LABELS,
        "metrics": scores,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
