"""
Baseline training procedure
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from utils.dataset import cifar_dataloader
from utils.optimizer import get_default_optim, decay_lr
from sklearn.decomposition import PCA
import models.convnext as cn
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
import numpy as np

AUTOMOBILE_IDX = 1  # CIFAR10, OOD-example
TURTLE_IDX = 93  # (94)
SUNFLOWER_IDX = 82 # (83) 


def main():
    _, cifar10_dataloader = cifar_dataloader(classes=10)
    _, cifar100_dataloader = cifar_dataloader(classes=100)


    convnext_22k = cn.convnext_base(pretrained=True, in_22k=True)

    # gather embeddings for each class
    cifar10_embeddings = [
        torch.empty((0, 1024)) for _ in range(10)
    ]
    cifar100_embeddings = [
        torch.empty((0, 1024)) for _ in range(100)
    ]
    with tqdm(cifar10_dataloader, unit="batch") as tepoch:
        for input, target in tepoch:
            print("")
            # embedding = convnext_22k.forward_features(input)
            cifar10_embeddings[target] = torch.cat()


if __name__ == "__main__":
    main()
