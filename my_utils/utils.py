

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, MetricCollection, Precision, Recall
from torchvision import transforms



def main_arg_parser():
    parser = argparse.ArgumentParser(description="Process script arguments")

    parser.add_argument('--wandb_sweep', action='store_true', default=None, help='If True, the script is run by wandb sweep')

    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
    parser.add_argument('--model', type=str, default='cnn', help='Model to use')

    return parser



def init_metrics(num_classes=10):
    return MetricCollection([
        Accuracy(task='multiclass', num_classes=num_classes, multiclass=True),
        # Precision(num_classes=num_classes, multiclass=True),
        # Recall(num_classes=num_classes, multiclass=True)
    ])


def isomorphic_data_augmentation(rotation=10, color_jitter=False) -> transforms.Compose:
    """
    Returns a torchvision.transforms.Compose object that performs data augmentation

    The data augmentation consists of:
        - Random rotation of the image by a maximum of `rotation` degrees
        - Random translation of the image by a maximum of 10% of the image size
        - Random horizontal flip with probability 0.5
        - Random vertical flip with probability 0.5
        - Random color jitter with brightness, contrast, saturation and hue all randomly changed by a maximum of 0.1
        - Normalization of the image to the range [-1, 1]

    Parameters
    ----------

    rotation : int
        Maximum number of degrees to rotate the image by
    
    color_jitter : bool
        If True, perform random color jitter on the image
    """

    T = transforms.Compose([
        transforms.RandomRotation(rotation),
        #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    T.transforms.append(transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip()], p=0.5))
    T.transforms.append(transforms.RandomApply(transforms=[transforms.RandomVerticalFlip()], p=0.5))

    if color_jitter:
        T.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    
    T.transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return T


def cifar_data_augmentation():
    return transforms.AutoAugmentPolicy.CIFAR10


def mix_data_augmentation():
    return transforms.AugMix()




def view_classify(img, logits):
    ''' 
    Function for viewing an image and it's predicted classes.
    '''

    img = img.view(1, 28, 28)
    ps = torch.exp(logits)
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()