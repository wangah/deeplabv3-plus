# DeepLabv3+

Final Project for DS4440. Reimplementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611)
with a modified ResNet-50 backbone.

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

### Download Dataset

#### gtFine

Download the Cityscapes files gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip [here](https://www.cityscapes-dataset.com/downloads/) and unzip into `./data`.

#### gtCoarse

Download gtCoarse.zip [here](https://www.cityscapes-dataset.com/downloads/) and
unzip into `./data`.

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
```

### Training

```bash
```

## Todo

- [ ] Use coarsely labeled images in training as well
- [ ] Track more metrics (Dice Score and iIoU)
- [ ] Investigate other loss functions (RMI)
- [ ] Experiment with Pytorch Lightning
