import math
import torch, gc
from torch import nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 对于多标记的计算指标
# 将多标记的结果转化为0，1
def predict(Outputs, threshold):
    sig = nn.Sigmoid()
    # pre = sig(Outputs)
    pre_label = sig(Outputs)
    pre = pre_label
    pre[pre > threshold] = 1
    pre[pre <= threshold] = 0
    return pre

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # gc.collect()
    # torch.cuda.empty_cache()

    y_a, y_b = y, y[index]
    y_new = lam * y + (1 - lam) * y[index, :]
    return mixed_x, y_new, y_a, y_b, lam

#####################################################################################################
# the loss of GDF
def class_mae(outputs, com_labels):
    '''
    for a drive loss based the bce loss, where l means -log or others loss function.
    The loss is an upper-bound of loss_unbiase_1.
    :param outputs: the presicted value with n*K size
    :param com_labels: the complementary label matrix, which is a one-hot vector for per instance
    :return: the loss value
    '''
    n, K = com_labels.size()[0], com_labels.size()[1]
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = -torch.sum(torch.log(sig_outputs + 1e-12) * pos_outputs, dim=1).mean()
    part_3 = -torch.sum(torch.log(1.0 - sig_outputs + 1e-12) * neg_outputs, dim=1).mean()
    ave_loss = part_1 + part_3
    return ave_loss