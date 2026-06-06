import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    macro_f1 = float(np.mean(f1_c))

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
        "per_label_f1": f1_c.astype(float).tolist(),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.05):
        scores = compute_scores(y_true, y_prob, float(threshold))
        if scores["micro_f1"] > best_f1:
            best_f1 = scores["micro_f1"]
            best_threshold = float(threshold)
    return best_threshold, best_f1


def load_annotations(path: str):
    with open(path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    id_to_labels = {}
    for split in ["train", "val", "test"]:
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
    parser = argparse.ArgumentParser(description="Train the clinical evidence branch for SEDRRG.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="mimic_cxr", choices=["iu_xray", "mimic_cxr"])

    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--vocab_threshold", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)

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

    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_head_lr", type=float, default=3e-4)
    parser.add_argument("--finetune_visual_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_pos_weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init_checkpoint", default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "clinical_evidence_best.pth"
    metrics_path = output_dir / "clinical_evidence_metrics.json"

    annotations, id_to_labels = load_annotations(args.ann_path)

    model_args = build_model_args(args)
    tokenizer = Tokenizer(model_args)
    model_args.vocab_size = tokenizer.get_vocab_size()
    model_args.pad_idx = int(tokenizer.pad_token_id)
    model_args.eos_idx = int(tokenizer.eos_token_id)
    model_args.bos_idx = int(tokenizer.cls_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {
        split: R2DataLoader(model_args, tokenizer, split=split, shuffle=(split == "train"))
        for split in ["train", "val", "test"]
    }

    train_labels = torch.stack([id_to_labels[str(item["id"])] for item in annotations["train"]], dim=0)
    positives = train_labels.sum(dim=0)
    negatives = train_labels.size(0) - positives
    pos_weight = (negatives / torch.clamp(positives, min=1.0)).clamp(
        min=1.0,
        max=args.max_pos_weight,
    ).to(device)

    base_model = R2GenModel(model_args, tokenizer).to(device)
    visual_extractor = base_model.visual_extractor

    ids, images, _, _ = next(iter(loaders["train"]))
    images = images.to(device)
    with torch.no_grad():
        feature_dim = pooled_visual_feature(visual_extractor, images).size(-1)

    classifier = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, len(CLINICAL_LABELS)),
    ).to(device)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        if "visual" in checkpoint:
            visual_extractor.load_state_dict(checkpoint["visual"], strict=False)
        if "head" in checkpoint:
            classifier.load_state_dict(checkpoint["head"], strict=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def set_visual_trainable(flag: bool):
        for parameter in visual_extractor.parameters():
            parameter.requires_grad = flag
        visual_extractor.train(flag)

    def train_epoch(optimizer, train_visual: bool):
        visual_extractor.train(train_visual)
        classifier.train()

        total_loss = 0.0
        steps = 0

        for ids, images, _, _ in loaders["train"]:
            images = images.to(device)
            labels = labels_for_ids(ids, id_to_labels, device)

            optimizer.zero_grad(set_to_none=True)

            if train_visual:
                features = pooled_visual_feature(visual_extractor, images)
            else:
                with torch.no_grad():
                    features = pooled_visual_feature(visual_extractor, images)

            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            steps += 1

        return total_loss / max(steps, 1)

    def evaluate(split: str):
        visual_extractor.eval()
        classifier.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for ids, images, _, _ in loaders[split]:
                images = images.to(device)
                labels = labels_for_ids(ids, id_to_labels, device)
                features = pooled_visual_feature(visual_extractor, images)
                logits = classifier(features)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)

    best_val_f1 = -1.0
    history = []

    set_visual_trainable(False)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.head_epochs + 1):
        loss = train_epoch(optimizer, train_visual=False)
        val_true, val_prob = evaluate("val")
        threshold, _ = select_threshold(val_true, val_prob)
        val_scores = compute_scores(val_true, val_prob, threshold)

        test_true, test_prob = evaluate("test")
        test_scores = compute_scores(test_true, test_prob, threshold)

        record = {
            "stage": "head",
            "epoch": epoch,
            "loss": loss,
            "threshold": threshold,
            "val": val_scores,
            "test": test_scores,
        }
        history.append(record)

        print(
            f"[head][epoch {epoch}] loss={loss:.4f} "
            f"threshold={threshold:.2f} "
            f"val_micro_f1={val_scores['micro_f1']:.4f} "
            f"test_micro_f1={test_scores['micro_f1']:.4f}",
            flush=True,
        )

        if val_scores["micro_f1"] > best_val_f1:
            best_val_f1 = val_scores["micro_f1"]
            torch.save(
                {
                    "visual": visual_extractor.state_dict(),
                    "head": classifier.state_dict(),
                    "threshold": threshold,
                    "label_names": CLINICAL_LABELS,
                    "model_args": vars(model_args),
                    "best_record": record,
                },
                best_path,
            )

    set_visual_trainable(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": visual_extractor.parameters(), "lr": args.finetune_visual_lr},
            {"params": classifier.parameters(), "lr": args.finetune_head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.finetune_epochs + 1):
        loss = train_epoch(optimizer, train_visual=True)
        val_true, val_prob = evaluate("val")
        threshold, _ = select_threshold(val_true, val_prob)
        val_scores = compute_scores(val_true, val_prob, threshold)

        test_true, test_prob = evaluate("test")
        test_scores = compute_scores(test_true, test_prob, threshold)

        record = {
            "stage": "finetune",
            "epoch": epoch,
            "loss": loss,
            "threshold": threshold,
            "val": val_scores,
            "test": test_scores,
        }
        history.append(record)

        print(
            f"[finetune][epoch {epoch}] loss={loss:.4f} "
            f"threshold={threshold:.2f} "
            f"val_micro_f1={val_scores['micro_f1']:.4f} "
            f"test_micro_f1={test_scores['micro_f1']:.4f}",
            flush=True,
        )

        if val_scores["micro_f1"] > best_val_f1:
            best_val_f1 = val_scores["micro_f1"]
            torch.save(
                {
                    "visual": visual_extractor.state_dict(),
                    "head": classifier.state_dict(),
                    "threshold": threshold,
                    "label_names": CLINICAL_LABELS,
                    "model_args": vars(model_args),
                    "best_record": record,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location="cpu")
    visual_extractor.load_state_dict(checkpoint["visual"], strict=False)
    classifier.load_state_dict(checkpoint["head"], strict=False)
    threshold = float(checkpoint["threshold"])

    test_true, test_prob = evaluate("test")
    final_scores = compute_scores(test_true, test_prob, threshold)

    result = {
        "threshold": threshold,
        "label_names": CLINICAL_LABELS,
        "final_test": final_scores,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Best checkpoint:", best_path)
    print("Metrics:", metrics_path)
    print("Final test:", final_scores)


if __name__ == "__main__":
    main()
