from torch.utils.data import DataLoader
from utils import dataset
from scipy.special import comb
import numpy as np
import torch


def choose(args):
    if args.dataset == "bookmark15":
        print('Data Preparation of bookmark15')
        file_name = ["./data/bookmark/bookmark15_data.csv", "./data/bookmark/bookmark15_label.csv",
                     "./data/bookmark/bookmark15_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 15
        input_dim = 2150
    elif args.dataset == "ml_tmc":
        print('Data Preparation of ml_tmc')
        file_name = ["./data/ml_tmc2007/ml_tmc2007_data.csv", "./data/ml_tmc2007/ml_tmc2007_label.csv",
                     "./data/ml_tmc2007/ml_tmc2007_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 22
        input_dim = 981

    return train_loader, test_loader, num_class, input_dim


def generate_uniform_comp_labels2(labels):
    K = np.size(labels, 1)
    n = np.size(labels, 0)
    cardinality = 2 ** K - 2
    number = np.array([comb(K, i+1) for i in range(K-1)])  # 1 to K-1, convert list to array
    frequency_dis = number / cardinality
    prob_dis = np.zeros(K - 1)  # array of K-1
    for i in range(K-1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

    random_n = np.random.uniform(0, 1, n)  # array: n
    mask_n = np.ones(n)  # n is the number of train_data
    comp_Y = np.zeros([n, K])  # complementary label matrix
    temp_num_comp_train_labels = 0  # save temp number of comp train_labels

    for j in range(n):
        # print(labels[j,:])
        for jj in range(K-1):
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_comp_train_labels = jj+1  # decide the number of complementary train_labels
                mask_n[j] = 0
        if temp_num_comp_train_labels > K - labels[j, :].sum():
            comp_Y[j, :] = 1 - labels[j, :]
        else:
            # 找到0的索引
            zero_indices = np.where(labels[j, :] == 0)[0]
            # 从0的索引中随机选择 temp_num_comp_train_labels 个
            selected_indices = np.random.choice(zero_indices, temp_num_comp_train_labels, replace=False)
            # 将选中的0标记为1
            comp_Y[j,selected_indices] = 1
    return comp_Y