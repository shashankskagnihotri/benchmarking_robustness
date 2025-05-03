# Disparity Estimation Benchmark

This repository provides tools and pre-trained models for benchmarking the robustness of disparity estimation.

---

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

---

## Preparation

### Pretrained Weights

1. Download the file `pretrain_weights.zip`
2. Place it under the directory `disparity_estimation/`
3. Unzip the contents

### Datasets

#### KITTI2015

1. Download `kitti2015.zip`
2. Place it under the directory `disparity_estimation/datasets/`
3. Unzip the contents

#### SceneFlow

To download and prepare the corrupted SceneFlow dataset, run:

```bash
cd datasets
./download_sceneflow.sh
./create_sceneflow.sh
```

---

## How to Use

### Model Zoo

You can load a pre-trained model by specifying the model name and the dataset it was last fine-tuned on:

```python
from disparity_estimation.evals import load_model

model = load_model(model_name='GWCNet-G', dataset='SceneFlow')
```

### Evaluation

#### Adversarial Attacks

```python
from disparity_estimation.evals import evaluate

model, results = evaluate(
    model_name='GWCNet-G',
    dataset='KITTI2015',
    retrieve_existing=False,
    threat_config='path/to/adv_attacks.yml',
)
```

- `retrieve_existing` is a boolean flag. If set to `True` and a matching evaluation exists in the benchmark, the cached result will be returned. Otherwise, the evaluation will be run.
- `adv_attacks.yml` should contain:
  - `threat model`: the name of the adversarial attack (supported: `'FGSM'`, `'PGD'`, `'APGD'`, `'BIM'`, `'CosPGD'`)
  - `iterations`: number of attack iterations
  - `epsilon`: permissible perturbation budget (ε)
  - `alpha`: step size of the attack (ϑ)
  - `lp norm`: the norm used to bound perturbation (`'Linf'` or `'L2'`)

#### 2D Common Corruptions

```python
from disparity_estimation.evals import evaluate

model, results = evaluate(
    model_name='GWCNet-G',
    dataset='KITTI2015',
    retrieve_existing=False,
    threat_config='path/to/2d_corruptions.yml',
)
```

- `retrieve_existing` works as described above.
- `2d_corruptions.yml` should contain:
  - `threat model`: the name of the corruption (supported: `'gaussian noise'`, `'shot noise'`, `'impulse noise'`, `'defocus blur'`, `'frosted glass blur'`, `'motion blur'`, `'zoom blur'`, `'snow'`, `'frost'`, `'fog'`, `'brightness'`, `'contrast'`, `'elastic'`, `'pixelate'`, `'jpeg'`)
  - `severity`: an integer from 1 to 5 indicating the corruption severity

---

### Supported Models and Datasets

| Model        | Supported Datasets     |
|--------------|------------------------|
| STTR         | SceneFlow, KITTI2015   |
| STTR-Light   | SceneFlow              |
| GWCNet-G     | SceneFlow, KITTI2015   |
| GWCNet-GC    | SceneFlow              |
| CFNet        | SceneFlow              |
