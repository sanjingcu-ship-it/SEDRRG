# SEDRRG

Structured Evidence-Guided Discrete Diffusion for Image-Conditioned Radiology Report Generation.

This repository contains the code used for the SEDRRG radiology report generation experiments. The active model is a discrete report-token diffusion model for image-conditioned radiology report generation.

## Main components

- Discrete report-token diffusion for chest X-ray report generation.
- Implementation-consistent MFSL training objective.
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

## Reproducibility

For environment setup, data layout, split inspection, training, testing, phrase-boost ablation, and clinical efficacy evaluation, see `REPRODUCIBILITY.md`.

## Training

Train on IU X-Ray:

    bash train_iu_xray.sh

Train on MIMIC-CXR:

    bash train_mimic_cxr.sh

The scripts are runnable examples. Adjust paths, batch size, and worker number according to your local hardware and dataset location.

Checkpoint selection is based on validation-set BLEU-4. Held-out test sets are used only for final evaluation.

## Testing

Test on IU X-Ray:

    bash test_iu_xray.sh

Test on MIMIC-CXR:

    bash test_mimic_cxr.sh


## Medical word/phrase boost ablation

The IU X-Ray configuration uses tokenizer-derived multi-token medical phrase boosting during sampling (`--sample_ngram_boost 1.5`). To reproduce the inference-only ablation requested during review, run the same checkpoint and decoding configuration with the neutral no-boost factor (`--sample_ngram_boost 1.0`):

    bash test_iu_xray_no_boost.sh

In `models/r2gen.py`, a non-positive `--sample_ngram_boost` value disables phrase-logit reweighting, and `1.0` is neutral. This ablation changes only the inference-time lexical prior; it does not retrain the model or change the tokenizer, test split, checkpoint, denoising steps, temperature, top-k setting, or random seed.

For MIMIC-CXR, the released tokenizer is word-level and does not retain multi-token phrase IDs, so the phrase-boost mechanism has no practical effect under the default MIMIC-CXR configuration.

## Denoising trace export

To export token-space reverse diffusion traces, add these flags to main_test.py:

    --export_denoising_trace
    --trace_output results/iu_xray_trace/denoising_trace.jsonl

The exported trace records report-token states and sampling statistics in token space. It should not be interpreted as image or pixel denoising. The active diffusion sampler does not use beam search.

Note: METEOR evaluation requires Java. The provided conda environment includes OpenJDK for this purpose.

## Clinical efficacy evaluation

Clinical efficacy is evaluated after report generation. First apply the public CheXbert labeler, or a compatible 14-observation label extractor, to generated and reference reports. The CheXbert labeler and its checkpoint are not redistributed in this repository and should be obtained from the original CheXbert release.

The expected CSV format is one identifier column followed by the standard 14 CheXbert observation columns in this order: No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices. Labels are binarized before metric computation, with positive=1 and negative, uncertain, or unmentioned labels treated as 0.

An example label table is provided at `examples/ce_labeled_example.csv`.

Run the default MIMIC-CXR CE evaluation with:

    python compute_ce.py

By default, `compute_ce.py` expects:

    results/mimic_cxr/res_labeled.csv
    results/mimic_cxr/gts_labeled.csv

Custom paths can be supplied with:

    python compute_ce.py --res_csv path/to/res_labeled.csv --gts_csv path/to/gts_labeled.csv

The script reports CheXbert-label-based micro-averaged precision, recall, and F1 using the extracted label vectors.
## Notes

- The active class is R2GenModel.
- R2GenModel1 is retained only for legacy compatibility.
- The active training entry point is main_train_repro_final.py.
- main_train.py is retained for backward compatibility with earlier R2Gen-style experiments.
- Model checkpoints, datasets, logs, and generated result files are intentionally excluded from the repository.
