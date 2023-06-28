# Nonconvex Network Slimming (Pytorch)

This repository is an extension of the repository of [Network Slimming (Pytorch)](https://github.com/Eric-mingjie/network-slimming), an official pytorch implementation of the following paper:
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  

It incorporates L_p, 0 < p < 1, and transformed L1 for nonconvex regularization of the channel scores. In addition, the dataset SVHN is available to train on. 

Citation:
```
@InProceedings{Liu_2017_ICCV,
    author = {Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui},
    title = {Learning Efficient Convolutional Networks Through Network Slimming},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
}

@inproceedings{bui2020nonconvex,
  title={Nonconvex Regularization for Network Slimming: Compressing CNNs Even More},
  author={Bui, Kevin and Park, Fredrick and Zhang, Shuai and Qi, Yingyong and Xin, Jack},
  booktitle={International Symposium on Visual Computing},
  pages={39--53},
  year={2020},
  organization={Springer}
}

@article{bui2021nonconvex,
    author={Bui, Kevin and Park, Fredrick and Zhang, Shuai and Qi, Yingyong and Xin, Jack},
    journal={IEEE Access}, 
    title={Improving Network Slimming with Nonconvex Regularization}, 
    year={2021},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/ACCESS.2021.3105366}
}
```


## Dependencies
torch v0.3.1, torchvision v0.2.0

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10`, `cifar100`, or `SVHN` The `arch` argument specifies the architecture to use: `vgg`,`resnet` or
`densenet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python main.py --dataset cifar10 --arch vgg --depth 19
```

## Train with Sparsity

The `reg` argument specifies which regularization to use: `L1`, `TL1`, `Lp`, `MCP`, or `SCAD`. The `a` argument specifies the nonconvex parameter. In particular, for Lp regularization, `a` has to have values strictly between 0 and 1; for TL1, `a` has to have values greater than 0.
```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --reg L1
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --reg Lp --a 0.5
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --reg TL1 --a 1.0
```

## Prune

```shell
python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19 --epochs 160
```
