# SEDRRG

Structured Evidence-Guided Discrete Diffusion for Image-Conditioned Radiology Report Generation.

This repository contains the code used for the SEDRRG radiology report generation experiments. The active model is a discrete report-token diffusion model for image-conditioned radiology report generation.

## Main components

- Discrete report-token diffusion for chest X-ray report generation.
- Code-aligned MFSL training objective.
- Medical token/phrase consistency, report-structure token consistency, and lightweight image-memory alignment.
- CheXbert-label-based clinical efficacy evaluation after generation.
- Optional token-space denoising trace export.

CheXbert is not used as a training loss. It is used only after report generation to extract clinical observation labels from generated and reference reports for clinical efficacy evaluation.

## Environment

Create the environment with:

    conda env create -f environment.yml
    conda activate sedrrg

If your local CUDA or PyTorch version differs, install a compatible PyTorch build for your GPU first and then install the remaining dependencies.

## Data

The IU X-Ray and MIMIC-CXR datasets are not redistributed in this repository. Please obtain them from their original data providers and follow their data-use agreements.

Expected local layout:

    data/
      iu_xray/
        images/
        annotation.json
      mimic_cxr/
        images/
        annotation.json

The annotation JSON files should follow the standard train, validation, and test split format used by the dataloader.

## Training

Train on IU X-Ray:

    bash train_iu_xray.sh

Train on MIMIC-CXR:

    bash train_mimic_cxr.sh

The scripts are runnable examples. Adjust paths, batch size, and worker number according to your local hardware and dataset location.

## Testing

Test on IU X-Ray:

    bash test_iu_xray.sh

Test on MIMIC-CXR:

    bash test_mimic_cxr.sh

## Denoising trace export

To export token-space reverse diffusion traces, add these flags to main_test.py:

    --export_denoising_trace
    --trace_output results/iu_xray_trace/denoising_trace.jsonl

The exported trace records report-token states and sampling statistics in token space. It should not be interpreted as image or pixel denoising.

## Clinical efficacy evaluation

Clinical efficacy is evaluated after report generation. First apply CheXbert or a compatible label extractor to generated and reference reports to obtain 14-observation label tables. Then run:

    python compute_ce.py

By default, compute_ce.py expects:

    results/mimic_cxr/res_labeled.csv
    results/mimic_cxr/gts_labeled.csv

The script maps uncertain labels to non-positive labels and computes multi-label clinical efficacy metrics using the extracted label vectors.

## Notes

- The active class is R2GenModel.
- R2GenModel1 is retained only for legacy compatibility.
- The active training entry point is main_train_repro_final.py.
- main_train.py is retained for backward compatibility with earlier R2Gen-style experiments.
- Model checkpoints, datasets, logs, and generated result files are intentionally excluded from the repository.
