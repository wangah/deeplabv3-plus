# DeepLabv3+ with Hierarchical Multi-Scale Attention

Final Project for DS4440. Reimplementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611) 
and [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821v1).

Folder structure and base classes were generated from
[pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor
Huang](https://github.com/victoresque).

## Resources

- [Cityscapes Dataset Paper](https://arxiv.org/pdf/1604.01685.pdf)
- [Cityscapes scripts](https://github.com/mcordts/cityscapesScripts)
- [TorchVision tutorial](https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb)
- [Learn OpenCV tutorial](https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/)

## Instructions

1. Download the Cityscapes files gtFine_trainvaltest.zip and
   leftImg8bit_trainvaltest.zip [here](https://www.cityscapes-dataset.com) and
   unzip.
2. Set an environment variable named `CITYSCAPES_DATASET` that points to the directory
   containing the unzipped images `export CITYSCAPES_DATASET=./data` (Optional).
3. Install the required packages `pip install -r requirements.txt`.
