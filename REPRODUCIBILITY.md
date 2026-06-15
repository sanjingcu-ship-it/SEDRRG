# Reproducibility notes for SEDRRG

This document summarizes repository-level information for reproducing the SEDRRG experiments after obtaining the IU X-Ray and MIMIC-CXR datasets from their original providers.

## Repository version

Use the fixed review tag reported in the manuscript Code Availability statement:

    git checkout v1.0-review

During local development, use the latest commit on main only if the review tag has been updated to point to the same commit.

## Environment

Create the environment with:

    conda env create -f environment.yml
    conda activate sedrrg

If the installed CUDA or PyTorch version differs from the local GPU driver, install a compatible PyTorch build first and then install the remaining dependencies from environment.yml.

## Data layout

Datasets are not redistributed in this repository. Prepare local folders in the following format or edit the paths in the shell scripts:

    data/
      iu_xray/
        images/
        annotation.json
      mimic_cxr/
        images/
        annotation.json

The annotation JSON should contain train, validation, and test splits in the format expected by the dataloader.

## Inspecting split counts and missing sections

Use the included utility to report split counts and section availability:

    python tools/inspect_annotation.py --ann_path data/iu_xray/annotation.json
    python tools/inspect_annotation.py --ann_path data/mimic_cxr/annotation.json

This is intended to document the exact local annotation file used for reproduction.

## Training

IU X-Ray:

    bash train_iu_xray.sh

MIMIC-CXR:

    bash train_mimic_cxr.sh

The training scripts select checkpoints using validation-set BLEU-4. Held-out test sets are reserved for final evaluation.

## Testing

IU X-Ray:

    bash test_iu_xray.sh

MIMIC-CXR:

    bash test_mimic_cxr.sh

The test scripts assume that training has produced the corresponding model_best.pth under the configured results directory. Pretrained checkpoints are not redistributed with this repository.

## Medical word/phrase boost ablation

IU X-Ray uses tokenizer-derived medical phrase boosting during inference in the default script. To disable this inference-only boost while keeping the same checkpoint, split, seed, decoding length, temperature, and top-k setting, run:

    bash test_iu_xray_no_boost.sh

## Clinical efficacy evaluation

Clinical efficacy is evaluated after report generation. Generated and reference reports should first be labeled by CheXbert or a compatible 14-observation label extractor. Then run:

    python compute_ce.py

By default, compute_ce.py expects:

    results/mimic_cxr/res_labeled.csv
    results/mimic_cxr/gts_labeled.csv

The script maps uncertain labels to non-positive labels and reports CheXbert-label-based micro precision, recall, and F1.

## Metric scripts

Textual metrics are computed through modules/metrics.py and the included pycocoevalcap utilities. The reported benchmark metrics depend on using the same tokenization, report section selection, split file, and evaluation script.

## Random seeds

The released training and testing scripts use the seed specified in each shell script. For exact reproduction, keep the same seed, preprocessing, tokenizer threshold, maximum sequence length, diffusion steps, decoding length, temperature, top-k value, and phrase-boost setting.

## Files intentionally not included

The following files are intentionally excluded because they are dataset-specific, run-specific, large, or governed by external data-use agreements:

- IU X-Ray and MIMIC-CXR images
- Local annotation or split files derived from restricted datasets
- Model checkpoints
- Training logs
- Generated reports
- CheXbert-labeled output CSV files
