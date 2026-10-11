# Vec2Null

PyTorch implementation of Vec2Null. The code structure, dataset construction, identity splits, and one-shot identity unlearning protocol follow [MetaUnlearn / ONE-SHUI](https://github.com/tdemin16/one-shui), the code for [Unlearning Personal Data from a Single Image](https://openreview.net/forum?id=VxC4PZ71Ym).

## Environment

Use an AMD GPU environment on Linux x86_64 with glibc 2.28 or newer and a compatible AMD driver. The dependencies are configured for ROCm 6.2.4, PyTorch 2.6.0, and Python 3.10–3.12; Python 3.12 is recommended. From the repository root, create a virtual environment and install [requirements-amd.txt](requirements-amd.txt):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-amd.txt
python -m pip check
```

The file pins the ROCm 6.2.4 builds of PyTorch and TorchVision together with the Python dependencies. Install the compatible AMD driver separately from these Python packages.

Prepare the alignment sources with that environment's Python:

```bash
python scripts/prepare_alignment.py
```

Check the command-line interface:

```bash
python vec2null.py --help
```

Run `git lfs install` before adding `.pth` weights to Git; their LFS rules are in `.gitattributes`. After cloning, run `git lfs pull` to download the weights.

## Data and protocol

This repository does not distribute datasets or protocol files. Obtain the data, annotations, and preparation instructions from the [MetaUnlearn repository](https://github.com/tdemin16/one-shui#datasets). Supported datasets are CelebA, CelebA-HQ, and MUFAC.

Use target checkpoints produced by MetaUnlearn, containing `args`, `splits`, and `state_dict`. Saved identity splits and support images are reused. For example, prepare a target model using:

```bash
python main.py --method pretrain --dataset celebahq --num_identities 20 --seed 1 --data_dir datasets --store_dir checkpoints
```

## UnoFace

[configs/vec2null.json](configs/vec2null.json) loads `weights/unoface_20ep.pth`. Paths in this configuration are relative to the configuration file; command-line arguments override its defaults.

Only the pretrained UnoFace encoder and inference components are provided. UnoFace training code and training protocols are not included.

UnoFace requires five-landmark face alignment. The preparation script retrieves the pinned [InsightFace](https://github.com/deepinsight/insightface/tree/0159b24d44cbc91abbd07b7a6fab0ba923a7fddf) alignment code. Supply the SCRFD `det_500m.onnx` detector from the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo) as `weights/det_500m.onnx`, or override `--reference-detector`.

## Usage

Train the meta-loss:

```bash
python vec2null.py meta-train --config configs/vec2null.json --target checkpoints/celebahq/vit_base_patch16_224/pretrain_20_1.pth --data-root datasets --output output/meta --seed 1
```

Create a request JSON with exactly one entry for each support image in the target checkpoint's unlearning protocol. `identity` is the dataset identity identifier, and `image` is an absolute image path. Support labels and identity tokens follow MetaUnlearn's annotated request setting.

```json
[
  {"identity": "123", "image": "/path/to/support/image.jpg"}
]
```

Apply unlearning:

```bash
python vec2null.py unlearn --config configs/vec2null.json --target checkpoints/celebahq/vit_base_patch16_224/pretrain_20_1.pth --artifact output/meta/meta-last.pth --request request.json --data-root datasets --output output/unlearned --seed 1
```

Evaluate a checkpoint:

```bash
python vec2null.py evaluate --target output/unlearned/unlearned.pth --data-root datasets --output output/evaluation --seed 1
```

For comparisons, evaluate all methods with this repository's shared evaluator and identical checkpoint splits. The loss-based MIA uses a fixed 100-tree Gini random forest with seed 0, rather than selecting an attacker on the forget set.

## Reference

Thomas De Min, Massimiliano Mancini, Stéphane Lathuilière, Subhankar Roy, and Elisa Ricci. *Unlearning Personal Data from a Single Image*. Transactions on Machine Learning Research, 2025. [Paper](https://openreview.net/forum?id=VxC4PZ71Ym) · [Code](https://github.com/tdemin16/one-shui).

```bibtex
@article{min2025unlearning,
  title={Unlearning Personal Data from a Single Image},
  author={Thomas De Min and Massimiliano Mancini and St{\'e}phane Lathuili{\`e}re and Subhankar Roy and Elisa Ricci},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=VxC4PZ71Ym}
}
```
