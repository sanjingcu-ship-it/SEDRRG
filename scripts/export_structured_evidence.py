import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path
from argparse import Namespace

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


def load_annotations(path: str):
    with open(path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    id_to_item = {}
    for split in ["train", "val", "test"]:
        if split not in annotations:
            continue
        for item in annotations[split]:
            id_to_item[str(item["id"])] = item

    return annotations, id_to_item


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
    parser = argparse.ArgumentParser(description="Export structured clinical evidence for SEDRRG.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--threshold", type=float, default=None)

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

    args = parser.parse_args()

    annotations, id_to_item = load_annotations(args.ann_path)
    model_args = build_model_args(args)

    tokenizer = Tokenizer(model_args)
    model_args.vocab_size = tokenizer.get_vocab_size()
    model_args.pad_idx = int(tokenizer.pad_token_id)
    model_args.eos_idx = int(tokenizer.eos_token_id)
    model_args.bos_idx = int(tokenizer.cls_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = R2GenModel(model_args, tokenizer).to(device)
    visual_extractor = base_model.visual_extractor

    loader_for_shape = R2DataLoader(model_args, tokenizer, split="val", shuffle=False)
    ids, images, _, _ = next(iter(loader_for_shape))
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

    evidence = {
        "label_names": CLINICAL_LABELS,
        "threshold": threshold,
        "splits": {},
    }

    with torch.no_grad():
        for split in ["train", "val", "test"]:
            if split not in annotations:
                continue

            loader = R2DataLoader(model_args, tokenizer, split=split, shuffle=False)
            rows = []

            for ids, images, _, _ in loader:
                images = images.to(device)
                features = pooled_visual_feature(visual_extractor, images)
                logits = classifier(features)
                probs = torch.sigmoid(logits).detach().cpu()

                for sample_id, prob in zip(ids, probs):
                    sample_id = str(sample_id)
                    item = id_to_item[sample_id]
                    prob_list = [float(value) for value in prob.tolist()]
                    hard_labels = [1 if value >= threshold else 0 for value in prob_list]
                    positive_names = [
                        name for name, value in zip(CLINICAL_LABELS, hard_labels) if value == 1
                    ]

                    rows.append(
                        {
                            "id": sample_id,
                            "study_id": item.get("study_id"),
                            "subject_id": item.get("subject_id"),
                            "clinical_probs": prob_list,
                            "clinical_labels": hard_labels,
                            "clinical_positive_names": positive_names,
                            "reference_labels": item.get("ce_labels"),
                            "reference_positive_names": item.get("ce_positive_names"),
                        }
                    )

            evidence["splits"][split] = rows

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print(f"Structured evidence saved to: {output_path}")


if __name__ == "__main__":
    main()
