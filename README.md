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

## Resources

- [Cityscapes Dataset Paper](https://arxiv.org/pdf/1604.01685.pdf)
- [Cityscapes scripts](https://github.com/mcordts/cityscapesScripts)

## Instructions

1. Download the Cityscapes files gtFine_trainvaltest.zip and
   leftImg8bit_trainvaltest.zip [here](https://www.cityscapes-dataset.com) and
   unzip.
2. Set an environment variable named `CITYSCAPES_DATASET` that points to the directory
   containing the unzipped images `export CITYSCAPES_DATASET=./data` (Optional).
3. Install the required packages `pip install -r requirements.txt`.

## Todo

- [ ] Investigate more types of data augmentation (Gaussian Blur, Color Augmentation)
- [ ] Use coarsely labeled images in training as well
- [ ] Track more metrics like Dice Score and iIoU
- [ ] Investigate other loss functions
- [ ] Hierarchical Multi-Scale Attention
- [ ] Create Custom Trainer
