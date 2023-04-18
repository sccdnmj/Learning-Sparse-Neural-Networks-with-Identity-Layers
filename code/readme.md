# Learning Sparsity Networks with Identity Layers

This repo contains the official implementations of Learning Sparsity Networks with Identity Layers.

# Requirements

Python3.6/Python3.7/Python3.8

```
pip install torch
pip install torchvision
pip install torchaudio
```

# Dataset

1. Download ImageNet from [the official website of ImageNet](https://image-net.org/).
2. For cifar datasets, it will be automatically downloaded.



# Plug-and-Play

Our CKA-SR is implemented in /CKA-SR/CKASR.py. As our CKA-SR is a plug-and-play regularization, you can plug it into your training code of pre-training, sparse training and network pruning methods.

Note that **you should refer to /CKA-SR/resnet.py and /CKA-SR/cifar_resnet.py to modify your network structure**, so that you could use CKA-SR in your code.



# Examples

We plug our CKA-SR into the Random Sparse Pruning method as an example. The code of RST method is implemented in https://github.com/vita-group/random_pruning. Our experimental settings follow this implementation.

To train a ResNet20 with sparsity of 0.95 on CIFAR-100 dataset , open the /examples/Random_pruning/CIFAR directory and run: 

```python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model cifar_resnet_32 --data cifar100 --epoch 160 --cka_weight 1e-04 --samples-to-cal 8 --batches-to-cal 5```

Note that cka_weight (the hyperparameter beta of our CKA-SR) is set to 1e-04, samples-to-cal (for calculating CKA-SR with several samples of each batch) is set to 8 and batches-to-cal (for calculating CKA-SR once out of several batches) is set to 5. You can change these arguments. 

Note that **if you want to use all the samples of each batch to calculate CKA-SR, please set samples-to-cal to -1. ** For example, if you want to use all the samples to calculate CKA-SR, you should run:

```python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model cifar_resnet_32 --data cifar100 --epoch 160 --cka_weight 1e-04 --samples-to-cal -1 --batches-to-cal 1```



To train a ResNet18 with sparsity of 0.95 on ImageNet dataset, open the /examples/Random_pruning/ImageNet directory and run:

```python $1multiproc.py --nproc_per_node 4 $1main.py --sparse_init ERK_plus --fc_density 1.0 --fix --fp16 --master_port FREEPORT -j 10 -p 500 --arch resnet18  -c fanin --label-smoothing 0.1 -b 192 --lr 0.4 --warmup 5 --epochs 100 --density 0.05 --static-loss-scale 256 $2 /path/to/imagenet --save ./save --cka-weight 1e-05```