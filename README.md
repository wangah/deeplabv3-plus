# DeepLabv3+

Final Project for DS4440 - Practical Neural Networks at Northeastern University
(Fall 2020).

Reimplementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611) with a
modified ResNet-50 backbone as specified in [DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf).

The main results can be viewed in `evaluate.ipynb`.

Folder structure and base classes were generated from
[pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor
Huang](https://github.com/victoresque).

## Papers

- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf)
- [Cityscapes Dataset Paper](https://arxiv.org/pdf/1604.01685.pdf)

## Resources

- [Cityscapes scripts](https://github.com/mcordts/cityscapesScripts)

## Usage

### Dataset

Download the following files from [Cityscapes](https://www.cityscapes-dataset.com/downloads/):
    - gtFine_trainvaltest.zip
    - leftImg8bit_trainvaltest.zip
    - gtCoarse.zip
    - leftImg8bit_trainextra.zip

Extract the files into `./data/` (or wherever you specify as the `data_dir` in `config.json`).
Make sure to use the fine annotations for the train and val sets and coarse
annotations for the train_extra set:

```text
data/
│
├── gtFine/
│   ├── train/
│   └── val/
│
├── gtCoarse/
│   └── train_extra/
│  
└── leftImg8bit/
    ├── train/
    ├── train_extra/
    ├── val/
    └── test/
```

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
# image
python test.py

# video
python test_video.py  # TODO
```

### Training

```bash
python train.py
```

## Todo

- [ ] Track more metrics (Dice Score and iIoU)
- [ ] Investigate other loss functions (RMI)
