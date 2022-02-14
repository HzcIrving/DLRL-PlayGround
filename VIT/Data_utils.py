#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""准备+Preprocessing Data"""

import logging

import torch

import Utils

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)

CONFIGS = {
    'ViT-B_16': Utils.get_b16_config(),
    # 'ViT-B_32': Utils.get_b32_config(),
    # 'ViT-L_16': Utils.get_l16_config(),
    # 'ViT-L_32': Utils.get_l32_config(),
    # 'ViT-H_14': Utils.get_h14_config(),
    # 'R50-ViT-B_16': Utils.get_r50_b16_config(),
    'testing': Utils.get_testing(),
}

def get_loader(args):
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test)

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)
    # if args.local_rank == 0:
    #     torch.distributed.barrier()

    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4) if testset is not None else None

    return train_loader, test_loader