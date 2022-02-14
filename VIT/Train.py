#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import logging
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import datetime
from datetime import timedelta

import torch
import torch.distributed as dist

from Data_utils import get_loader
from Data_utils import CONFIGS
from Model import VITransModel
from Utils import WarmupCosineSchedule,WarmupLinearSchedule
from Utils import set_seed, AverageMeter, simple_accuracy, model_save

from tensorboardX import SummaryWriter

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

"""Config"""
class VITConfig:
    log_dir = "./TB_log/"
    dataset = "cifar10" # "cifar100"
    model_type = "ViT-B_16"
    pretrained_dir = "./Pretrained/imagenet21k_ViT-B_16.npz" # 预训练模型存放位置
    save_dir = "./Model/"
    record_algo = "Pretrained_VIT_Cifar10_ViTB16_"
    test_cycles = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    decay_type = "cosine" #  "cosine", "linear" 决定了学习率Scheduler类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TB_log = True

    img_size = 224
    train_batch_size = 64 #512
    eval_batch_size = 32 #64
    eval_every = 100 # Run prediction on validation set every so many steps.
    learning_rate = 3e-2 # SGD起始学习率
    weight_decay = 0 #
    num_steps = 10000 # Total number of training epochs to perform.
    warmup_steps = 500 # 开始的Warmup Step数
    max_grad_norm = 1.0

    local_rank = -1 # local_rank for distributed training on gpus
    seed = 42
    gradient_accumulation_steps = 1 # Number of updates steps to accumulate before performing a backward/update pass.


"""Model Valid Process"""
def valid(args,model,writer,test_loader,global_step):
    """
    :param args: 参数Config
    :param model:  需验证模型
    :param writer:  TB写入
    :param test_loader:  测试数据集
    :param global_step:  全局step
    :return:
    """
    # Validation
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [],[]
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    global_eval_step = 0
    for step, batch in enumerate(epoch_iterator):
        global_eval_step += 1
        batch = tuple(t.to(args.device) for t in batch)
        x,y = batch
        with torch.no_grad():
            logits = model(x)[0]
            eval_loss = loss_fct(logits,y)
            eval_losses.update(eval_loss.item()) #滑动平均
            preds = torch.argmax(logits,dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            # append在后面
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
        writer.add_scalar("Train/loss", scalar_value=eval_losses.val, global_step=global_eval_step)

    all_preds, all_label = all_preds[0], all_label[0]
    # all_preds: numpy.array; all_label: numpy.array;
    accuracy = simple_accuracy(all_preds,all_label)

    writer.add_scalar("test/accuracy",scalar_value=accuracy,global_step=global_step)

    return accuracy

"""Model Training Process"""
def train(args=VITConfig()):
    """
    :param args:
     - log_dir
    """
    # 模型准备
    pretrained_model_config = CONFIGS[args.model_type]
    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VITransModel(pretrained_model_config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(device=args.device)
    num_params = count_parameters(model)

    if args.TB_log:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(logdir=args.log_dir + args.record_algo + args.test_cycles)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 1. DATA准备
    train_loader, test_loader = get_loader(args)

    # 2. 准备优化器以及Scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.learning_rate, # init lr
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    t_total = args.num_steps # Total time steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # 3. Training
    model.zero_grad()
    set_seed(args.seed)
    losses = AverageMeter()
    global_step = 0
    best_acc = 0
    while True:
        model.train()

        # 一个数据迭代器
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x,y = batch # XData, YLabel
            loss = model.forward(x,y)
            loss.backward()

            if (step+1)%args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Print Training Info
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                writer.add_scalar("Train/loss",scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("Train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                # Valid ...
                if global_step % args.eval_every == 0:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        best_acc = accuracy
                        model_save(args.record_algo+args.test_cycles,model)
                    model.train()

                if global_step % t_total == 0:
                    break

        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    print("==="*30)
    print("Best Accuracy: \t%f" % best_acc)
    print("End Training!")
    print("==="*30)


if __name__ == "__main__":
    train()
    # all_preds = []
    # all_labels = []
    #
    # all_pred = torch.tensor([1,0,1,1,0,1])
    # all_label = torch.tensor([1,1,1,1,1,1])
    #
    # all_preds.append(all_pred)
    # all_labels.append(all_label)
    # print(all_preds)
    # all_preds[0] = np.append(all_preds[0],all_label,axis=0)
    # all_labels[0] = np.append(all_labels[0],all_pred,axis=0)
    # print(type(all_preds[0]))
    # print(type(all_labels[0]))
    # acc = simple_accuracy(all_preds[0],all_labels[0])
    # print(acc)