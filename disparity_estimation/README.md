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

1. Download the file [`pretrain_weights.zip`](https://data.dws.informatik.uni-mannheim.de/machinelearning/robustness_benchmarking/disparity_estimation/pretrained_weights.zip)
2. Unzip the contents

### Datasets

#### KITTI2015

1. Download the KITTI 2015 dataset from [KITTI Scene Flow Benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
2. After unzipping, ensure the contents include `training/` and `testing/` directories. Place them under `datasets/`.
3. Then run the following commands:

```bash
cd datasets
./create_kitti2015.sh
```

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
from dispbench.evals import load_model

model = load_model(model_name='GWCNet-G', dataset='SceneFlow')
```

### Evaluation

#### Adversarial Attacks

```python
from dispbench.evals import evaluate

model, results = evaluate(
    model_name='GWCNet-G',
    dataset='KITTI2015',
    retrieve_existing=False,
    threat_config='configs/adv_attacks.yml',
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
from dispbench.evals import evaluate

model, results = evaluate(
    model_name='GWCNet-G',
    dataset='KITTI2015',
    retrieve_existing=False,
    threat_config='configs/2d_corruptions.yml',
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
