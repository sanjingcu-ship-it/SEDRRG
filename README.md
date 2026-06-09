# SEDRRG

Official implementation of **SEDRRG: Structured Evidence-guided Discrete Diffusion for Radiology Report Generation**.

## Overview

SEDRRG is a radiology report generation framework that combines visual encoding, structured clinical evidence, and discrete diffusion-based text generation.

This repository provides:

- report generation training and evaluation;
- MIMIC-CXR clinical efficacy evaluation with 14 labels;
- structured clinical evidence extraction;
- reproducible configuration templates for IU X-Ray and MIMIC-CXR.

## Repository structure

```text
SEDRRG/
├── configs/
│   ├── iu_xray.yaml
│   ├── mimic_cxr.yaml
│   └── mimic_cxr_clinical_evidence.yaml
├── scripts/
│   ├── train_report_generation.py
│   ├── evaluate_report_generation.py
│   ├── train_clinical_evidence.py
│   ├── evaluate_clinical_efficacy.py
│   ├── evaluate_report_clinical_efficacy.py
│   └── export_structured_evidence.py
├── models/
├── modules/
├── pycocoevalcap/
├── examples/
└── docs/
```

## Data

This repository does not include IU X-Ray or MIMIC-CXR images or reports. Please obtain the datasets from their official sources and prepare annotations following `examples/sample_annotation_format.json`.

See `docs/data_preparation.md` for details.

## Training report generation

```bash
python scripts/train_report_generation.py \
  --image_dir /path/to/images \
  --ann_path /path/to/annotation.json \
  --dataset_name mimic_cxr \
  --max_seq_length 100 \
  --threshold 10 \
  --batch_size 16 \
  --num_diffusion_steps 9 \
  --sample_max_len 60
```

## Evaluating report generation

```bash
python scripts/evaluate_report_generation.py \
  --image_dir /path/to/images \
  --ann_path /path/to/annotation.json \
  --dataset_name mimic_cxr \
  --max_seq_length 100 \
  --threshold 10 \
  --sample_max_len 60 \
  --resume /path/to/checkpoint.pth
```

## Training the clinical evidence branch

```bash
python scripts/train_clinical_evidence.py \
  --image_dir /path/to/mimic-cxr-jpg \
  --ann_path /path/to/mimic_annotation_with_ce_labels.json \
  --output_dir outputs/mimic_cxr_clinical_evidence \
  --dataset_name mimic_cxr \
  --visual_extractor_pretrained
```

## Clinical efficacy evaluation

```bash
python scripts/evaluate_clinical_efficacy.py \
  --image_dir /path/to/mimic-cxr-jpg \
  --ann_path /path/to/mimic_annotation_with_ce_labels.json \
  --checkpoint outputs/mimic_cxr_clinical_evidence/clinical_evidence_best.pth \
  --output_path outputs/mimic_cxr_clinical_evidence/test_metrics.json \
  --split test \
  --visual_extractor_pretrained
```

## Exporting structured evidence

```bash
python scripts/export_structured_evidence.py \
  --image_dir /path/to/mimic-cxr-jpg \
  --ann_path /path/to/mimic_annotation_with_ce_labels.json \
  --checkpoint outputs/mimic_cxr_clinical_evidence/clinical_evidence_best.pth \
  --output_path outputs/mimic_cxr_clinical_evidence/structured_evidence.json \
  --visual_extractor_pretrained
```


## Medical word/phrase boost ablation

The default report-generation evaluation uses `--sample_ngram_boost 1.5`. To disable the medical word/phrase boost for the inference-only ablation, use `--sample_ngram_boost 1.0` while keeping the checkpoint, tokenizer, split, and other sampling settings unchanged.

## Report-level clinical efficacy evaluation

```bash
python scripts/evaluate_report_clinical_efficacy.py \
  --pred_path /path/to/generated_reports.json \
  --ref_path /path/to/reference_reports_or_labels.json \
  --output_path outputs/report_clinical_efficacy.json
```

## License

Please check the license before using this code.
