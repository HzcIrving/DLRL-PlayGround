#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""训练脚本
# 读取预训练权重--需要删除分类相关的权重，自己的分类类别数与预训练
                 分类类别数不同。
# freeze_layers--冻结除了最后一个FC层以外的layer，只训练最后一层。
"""

import os
import sys
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# Data Transform
from torchvision import transforms
import datetime

from Utils import MyDataSet
from Model import swin_tiny_patch4_window7_224 as create_model
from Utils import get_cifar10_loader

class cifar10_train_config:
    log_dir = "../../../DeepLearning_PlayGround/SwinTransformer/TB_log/"
    dataset = "cifar10" # "cifar100"
    save_dir = "./Model/"
    record_algo = "Pretrained_VIT_Cifar10_ViTB16_"
    weights = './weights/swin_tiny_patch4_window7_224.pth'
    test_cycles = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # TODO 自己的数据路径
    my_data_path = './data/flower_photos'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TB_log = True
    freeze_layers = False

    num_classes = 1000
    epochs = 100
    train_batch_size = 64 #512 # 8
    eval_batch_size = 32 #64 # 8
    lr = 0.0001
    img_size = 224
    eval_every = 10

def train_one_epoch(model,optimizer,data_loader,device,epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step,data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred,labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,  accu_loss.item() / (step + 1), accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def eval(model,data_loader,device,epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1),accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def main(args=cifar10_train_config()):
    device = args.device
    if args.TB_log:
        os.makedirs(args.log_dir,exist_ok=True)
        tb_writer = SummaryWriter(log_dir=args.log_dir + args.record_algo + args.test_cycles)

    # 权重文件夹
    if os.path.exists("../../../DeepLearning_PlayGround/SwinTransformer/Models") is False:
        os.makedirs("../../../DeepLearning_PlayGround/SwinTransformer/Models")

    # ====== Cifar10 =======
    train_loader,val_loader = get_cifar10_loader(args)

    # ====== Own Dataset ======
    # TODO
    # -------------------------

    # Model Create
    model = create_model(num_classes=args.num_classes).to(device)

    # 预训练权重读取
    if args.weights!="":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights,map_location=device)["model"]
        # 删除分类相关权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
        # print(model.load_state_dict(weights_dict, strict=False))

    # 是否采用冻结其他权重，只训练最后一个FC层来加速训练
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除分类头都要冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    # 存储最好的权重
    best_acc = 0.0

    for epoch in range(args.epochs):
        # train:
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # eval
        if epoch % args.eval_every == 0:
            val_loss, val_acc = eval(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

        if args.TB_log:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            torch.save(model.state_dict(), "./Models/model-{}.pth".format(epoch))

if __name__ == "__main__":
    main()