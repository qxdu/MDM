

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import time
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import sys
import random
sys.path.append("../utils/")

from mds import *
from plot import *

from scipy import stats



def trim_list(data):
    sorted_data = sorted(data)  
    trimmed_data = sorted_data[5:-5] 
    return trimmed_data

data1, data2, data3 = [], [], []
K = 100

for i in range(0, K):
    natural_path = "../dataset/diffusion_promoter/sequence_data.txt"
    mdm_samples_path = "../checkpoints/ecoli_50/mdm/samples/seed_{}.txt".format(i)
    ddsm_samples_path = "../checkpoints/ecoli_50/ddsm/samples/seed_{}.txt".format(i)
    wgan_samples_path = "../checkpoints/ecoli_50/wgan/samples/seed_{}.txt".format(i)
    
    avg_dis_mdm = inner_levenshtein_distance_calculation(mdm_samples_path)
    avg_dis_ddsm = inner_levenshtein_distance_calculation(ddsm_samples_path)
    avg_dis_wgan = inner_levenshtein_distance_calculation(wgan_samples_path)
    print("Epoch {}: [{}, {}, {}]".format(i, avg_dis_mdm, avg_dis_ddsm, avg_dis_wgan))
    
    data1.append(avg_dis_mdm)
    data2.append(avg_dis_ddsm)
    data3.append(avg_dis_wgan)

data1 = trim_list(data1)
data2 = trim_list(data2)
data3 = trim_list(data3)

stat, p_value = stats.wilcoxon(np.array(data1) - np.array(data2))
print(p_value)

distance = data1 + data2 + data3
labels = ["mdm"] * len(data1) + ["ddsm"] * len(data2) + ["wgan"] * len(data3)

df_concat = pd.DataFrame({"distance": distance, "labels": labels})

plot_boxplot("./results/boxplot_diff.png", df_concat)

