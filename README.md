# BNTT-Batch-Normalization-Through-Time

This repository contains the source code associated with [arXiv preprint arXiv:2010.01729][arXiv preprint arXiv:2010.01729]

Accepted to Frontiers in Neuroscience (2021)

[arXiv preprint arXiv:2010.01729]: https://arxiv.org/abs/2010.01729

## Introduction

Spiking Neural Networks (SNNs) have recently emerged as an alternative to deep learning owing to sparse, asynchronous and binary event (or spike) driven processing, that can yield huge energy efficiency benefits on neuromorphic hardware. However, training high-accuracy and low-latency SNNs from scratch suffers from non-differentiable nature of a spiking neuron. To address this training issue in SNNs, we revisit batch normalization and propose a temporal Batch Normalization Through Time (BNTT) technique. Most prior SNN works till now have disregarded batch normalization deeming it ineffective for training temporal SNNs. Different from previous works, our proposed BNTT decouples the parameters in a BNTT layer along the time axis to capture the temporal dynamics of spikes. The temporally evolving learnable parameters in BNTT allow a neuron to control its spike rate through different time-steps, enabling low-latency and low-energy training from scratch. We conduct experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet and event-driven DVS-CIFAR10 datasets. BNTT allows us to train deep SNN architectures from scratch, for the first time, on complex datasets with just few 25-30 time-steps. We also propose an early exit algorithm using the distribution of parameters in BNTT to reduce the latency at inference, that further improves the energy-efficiency.


## Prerequisites
* Ubuntu 18.04    
* Python 3.6+    
* PyTorch 1.5+ (recent version is recommended)     
* NVIDIA GPU (>= 12GB)        

## Getting Started

### Installation
* Configure virtual (anaconda) environment
```
conda create -n env_name python=3.7
source activate env_name
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```


## Training and testing

* We provide VGG9/VGG11 architectures on CIFAR10/CIAR100 datasets
* ```train.py```: code for training  
* ```model.py```: code for VGG9/VGG11 Spiking Neural Networks with BNTT  
* ```utill.py```: code for accuracy calculation / learning rate scheduler

*  Run the following command for VGG9 SNN on CIFAR10

```
python train.py --num_steps 25 --lr 0.3 --arch 'vgg9' --dataset 'cifar10' --batch_size 256 --leak_mem 0.95 --num_workers 4 --num_epochs 100
```

*  Run the following command for VGG11 SNN on CIFAR100

```
python train.py --num_steps 30 --lr 0.3 --arch 'vgg11' --dataset 'cifar100' --batch_size 128 --leak_mem 0.99 --num_workers 4 --num_epochs 100
```


## Citation
 
Please consider citing our paper:
 ```
 @article{kim2020revisiting,
  title={Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch},
  author={Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2010.01729},
  year={2020}
}
 ```
 
 

