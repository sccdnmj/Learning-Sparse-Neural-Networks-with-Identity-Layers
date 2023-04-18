# Examples

We plug our CKA-SR into the Random Sparse Pruning method as an example. The code of RST method is implemented in https://github.com/vita-group/random_pruning. Our experimental settings follow this implementation.

To train a ResNet20 with sparsity of 0.95 on CIFAR-100 dataset , open the CIFAR directory and run: 

```python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model cifar_resnet_32 --data cifar100 --epoch 160 --cka_weight 1e-04 --samples-to-cal 8 --batches-to-cal 5```

Note that cka_weight (the hyperparameter beta of our CKA-SR) is set to 1e-04, samples-to-cal (for calculating CKA-SR with several samples of each batch) is set to 8 and batches-to-cal (for calculating CKA-SR once out of several batches) is set to 5. You can change these arguments. 

Note that **if you want to use all the samples of each batch to calculate CKA-SR, please set samples-to-cal to -1. ** For example, if you want to use all the samples to calculate CKA-SR, you should run:

```python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model cifar_resnet_32 --data cifar100 --epoch 160 --cka_weight 1e-04 --samples-to-cal -1 --batches-to-cal 1```



To train a ResNet18 with sparsity of 0.95 on ImageNet dataset, open the ImageNet directory and run:

```python $1multiproc.py --nproc_per_node 4 $1main.py --sparse_init ERK_plus --fc_density 1.0 --fix --fp16 --master_port FREEPORT -j 10 -p 500 --arch resnet18  -c fanin --label-smoothing 0.1 -b 192 --lr 0.4 --warmup 5 --epochs 100 --density 0.05 --static-loss-scale 256 $2 /path/to/imagenet --save ./save --cka-weight 1e-05```