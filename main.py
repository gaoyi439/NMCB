import argparse
import os

import torch.nn as nn
import torch
import numpy as np
from torch.backends import cudnn
import random

from utils.metrics import OneError, Coverage, HammingLoss, RankingLoss, AveragePrecision
from utils.models import linear, MLP
from utils.utils_algo import adjust_learning_rate, predict, class_mae, AverageMeter, mixup_data
from utils.utils_data import choose

parser = argparse.ArgumentParser(description='PyTorch implementation of TPAMI 2025 for NMCB')
parser.add_argument('--dataset', default='bookmark15', type=str)
parser.add_argument('--num-class', default=15, type=int, help='number of classes')
parser.add_argument('--input-dim', default=2150, type=int, help='number of features')
parser.add_argument('--fold', default=9, type=int, help='fold-th fold of 10-cross fold')
parser.add_argument('--model', default="linear", type=str, choices=['MLP', 'linear'])
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--lo', default="NMCB", type=str)
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--the', default=0.5, type=float, help='seed for initializing training. ')
parser.add_argument('--lam', default=1, type=float)
parser.add_argument('--alpha', default=0.9, type=float)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()


def main():
    print(args)

    cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # make data
    train_loader, test_loader, args.num_class, args.input_dim = choose(args)

    # choose model
    if args.model == "linear":
        model = linear(input_dim=args.input_dim, output_dim=args.num_class)
        model_1 = linear(input_dim=args.num_class, output_dim=args.num_class)
    elif args.model == "MLP":
        model = MLP(input_dim=args.input_dim, hidden_dim=500, output_dim=args.num_class)
        model_1 = linear(input_dim=args.num_class, output_dim=args.num_class)

    model = model.to(device)
    model_1 = model_1.to(device)

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)

    print("start training")

    best_av = 0
    save_table = np.zeros(shape=(args.epochs, 7))
    experiment_path = 'experiment/{ds}/'.format(ds=args.dataset)
    result_path = 'result/{ds}/'.format(ds=args.dataset)
    data_name = "{ds}_{md}_{M}_lr{lr}_wd{wd}_fold{fd}_alph{al}_lam{lam}".format(ds=args.dataset, md=args.lo,
                                                                                  M=args.model, lr=args.lr, wd=args.wd,
                                                                                  fd=args.fold, al=args.alpha, lam=args.lam)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, GDF, consist = train(train_loader, model, optimizer, args, epoch, optimizer_1, model_1)
        t_hamm, t_one_error, t_converage, t_rank, t_av_pre = validate(test_loader, model, args)
        print("Epoch:{ep}, Tr_loss:{tr}, T_hamm:{T_hamm}, T_one_error:{T_one_error}, T_con:{T_con}, "
              "T_rank:{T_rank}, T_av:{T_av}, GDF:{gdf}, sist:{sist}".format(ep=epoch, tr=train_loss, T_hamm=t_hamm, T_one_error=t_one_error,
                                                    T_con=t_converage, T_rank=t_rank, T_av=t_av_pre, gdf=GDF, sist=consist))
        save_table[epoch, :] = epoch + 1, train_loss, t_hamm, t_one_error, t_converage, t_rank, t_av_pre

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        np.savetxt(result_path+data_name+'.csv', save_table, delimiter=',', fmt='%1.4f')
        # save model
        if t_av_pre > best_av:
            best_av = t_av_pre
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
            torch.save(model.state_dict(), experiment_path+data_name+'.tar')



def train(train_loader, model, optimizer, args, epoch, optimizer_1, model_1):
    # global loss
    model.train()
    train_loss = 0
    sig = nn.Sigmoid()
    loss_GDF = AverageMeter()
    loss_consist = AverageMeter()
    loss_mixup = AverageMeter()

    for i, (images, img2, _, com_labels, index) in enumerate(train_loader):
        images, com_labels, img2 = images.to(device), com_labels.to(device), img2.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        optimizer_1.zero_grad()

        # mixup
        img_mixup, label_mixup, _, _, lam = mixup_data(img2, (1. - com_labels).clone(), args.alpha, use_cuda)
        outputs_1 = model(img_mixup)
        outputs_co = model_1(label_mixup.clone())
        out_co_pro = sig(outputs_co)

        loss_fn = nn.MSELoss()
        consist_loss1 = loss_fn(sig(outputs_1), out_co_pro)

        # mian loss: GDF
        loss_1 = class_mae(outputs, com_labels)

        # dynamic parameters
        dy = min((epoch / 200) * args.lam, args.lam)
        loss = loss_1 + dy * consist_loss1

        loss_GDF.update(loss_1.item())
        loss_mixup.update(consist_loss1.item())


        loss.backward()
        optimizer.step()
        optimizer_1.step()

        train_loss = train_loss + loss.item()

    print("GDF:{gdf}, sis:{sis}, mixup:{mix}".format(gdf=loss_GDF.avg, sis=loss_consist.avg, mix=loss_mixup.avg))
    return train_loss / len(train_loader), loss_GDF.avg, loss_consist.avg


# test the results
def validate(test_loader, model, args):
    with torch.no_grad():
        model.eval()
        sig = nn.Sigmoid()
        t_one_error = 0
        t_converage = 0
        t_hamm = 0
        t_rank = 0
        t_av_pre = 0

        for data, _, targets, _, _ in test_loader:
            images, targets = data.to(device), targets.to(device)
            output = model(images)
            pre_output = sig(output)
            pre_label = predict(output, args.the)

            t_one_error = t_one_error + OneError(pre_output, targets)
            t_converage = t_converage + Coverage(pre_output, targets)
            t_hamm = t_hamm + HammingLoss(pre_label, targets)
            t_rank = t_rank + RankingLoss(pre_output, targets)
            t_av_pre = t_av_pre + AveragePrecision(pre_output, targets)

    return t_hamm/len(test_loader), t_one_error/len(test_loader), t_converage/len(test_loader), t_rank/len(test_loader), \
        t_av_pre/len(test_loader)


if __name__ == '__main__':

    # learning rate for the linear model
    lr_1e_1 = ["yeast", "delicious15", "Corel16k15", "rcv1_15", "rcv2_15", "rcv3_15", "rcv4_15", "rcv5_15"]
    lr_1e_2 = ["mediamill15", "scene", "ml_tmc"]
    lr_1e_3 = ["ml_eurlex_dc15", "ml_eurlex_sm15", "bookmark15"]

    # ten-cross-fold
    for fd in range(10):
        args = parser.parse_args()

        if args.dataset in lr_1e_1:
            args.lr = 0.1
        elif args.dataset in lr_1e_2:
            args.lr = 0.01
        else:
            args.lr = 0.001

        args.fold = fd

        print("Data:{ds}, model:{model}, lr:{lr}, wd:{wd}, fold:{fd}, loss:{lo}".format(ds=args.dataset, model=args.model, lr=args.lr, wd=args.wd, fd=args.fold, lo=args.lo))
        main()

