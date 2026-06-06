# Data Preparation

This repository does not distribute IU X-Ray or MIMIC-CXR data.

Users should download the datasets from their official sources and prepare
annotation files following `examples/sample_annotation_format.json`.

## Annotation fields

Each sample should contain:

- `id`: unique study identifier.
- `image_path`: relative image path or a list of relative image paths.
- `report`: reference report text.
- `ce_labels`: optional 14-dimensional clinical efficacy label vector for MIMIC-CXR.
- `ce_positive_names`: optional list of positive clinical labels.

## Clinical labels

The clinical efficacy branch uses the following 14 labels:

1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices
