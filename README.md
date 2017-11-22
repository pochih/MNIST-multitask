## MNIST-multitask
Reproduce ICLR2018 under-reviewed paper [MULTI-TASK LEARNING ON MNIST IMAGE DATASETS](https://openreview.net/pdf?id=S1PWi_lC-)

the paper argues that pre-train network with MNIST-like dataset can boost performance

### results
dataset | single-task | F+N | single-task (paper reported) | F+N (paper reported)
--- | --- | --- | --- | --- 
MNIST | 0.996 | 0.9956 | 0.9956 | 0.9971
FashionMNIST | 0.9394 | 0.942 | 0.9432 | 0.9520

### discussion
in my reproduction, FashionMNIST performs better with MNIST+FashionMNIST pre-trained first

but MNIST doesn't enjoy the benefits of pre-training.

The bias between reproduction and paper can result from preprocess of data.
