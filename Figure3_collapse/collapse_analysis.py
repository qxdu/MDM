import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import random
import numpy as np
import pandas as pd
from scipy import stats

from tqdm import tqdm
from matplotlib import pyplot as plt

import os, sys
sys.path.append("../utils/")

from mds import *
from plot import *


## diversity plot

data1, data2, data3 = [], [], []
for i in range(1, 60 + 1):
    natural_path = "../dataset/diffusion_promoter/sequence_data.txt"
    mdm_samples_path = "./samples/mdm_60epoch/sample_ep{}_s0_num_10000.txt".format(i)
    ddsm_samples_path = "./samples/ddsm_60epoch/epoch_{}.txt".format(i)
    wgan_samples_path = "./samples/wgan_60epoch/sample_ep{}_s0_num_10000.txt".format(i)
    
    avg_dis_mdm = inner_levenshtein_distance_calculation(mdm_samples_path)
    avg_dis_ddsm = inner_levenshtein_distance_calculation(ddsm_samples_path)
    avg_dis_wgan = inner_levenshtein_distance_calculation(wgan_samples_path)
    print("Epoch {}: [{}, {}, {}]".format(i, avg_dis_mdm, avg_dis_ddsm, avg_dis_wgan))
    
    data1.append(avg_dis_mdm)
    data2.append(avg_dis_ddsm)
    data3.append(avg_dis_wgan)

plot_diversity(data1, data2, data3, "./results/Supp_diversity.png")


## cg content

data1, data2, data3 = [], [], []
for i in range(1, 60 + 1):
    natural_path = "../dataset/diffusion_promoter/sequence_data.txt"
    mdm_samples_path = "./samples/mdm_60epoch/sample_ep{}_s0_num_10000.txt".format(i)
    ddsm_samples_path = "./samples/ddsm_60epoch/epoch_{}.txt".format(i)
    wgan_samples_path = "./samples/wgan_60epoch/sample_ep{}_s0_num_10000.txt".format(i)
    
    cg_mdm = inner_CG_calculation(mdm_samples_path)
    cg_ddsm = inner_CG_calculation(ddsm_samples_path)
    cg_wgan = inner_CG_calculation(wgan_samples_path)
    print("Epoch {}: [{}, {}, {}]".format(i, cg_mdm, cg_ddsm, cg_wgan))
    
    data1.append(cg_mdm)
    data2.append(cg_ddsm)
    data3.append(cg_wgan)

plot_cg_content(data1, data2, data3, "./results/Supp_GC.png")


## poly calculation

data1, data2, data3 = [], [], []
for i in range(1, 60 + 1):
    natural_path = "../dataset/diffusion_promoter/sequence_data.txt"
    mdm_samples_path = "./samples/mdm_60epoch/sample_ep{}_s0_num_10000.txt".format(i)
    ddsm_samples_path = "./samples/ddsm_60epoch/epoch_{}.txt".format(i)
    wgan_samples_path = "./samples/wgan_60epoch/sample_ep{}_s0_num_10000.txt".format(i)
    
    poly_mdm = inner_poly_calculation(mdm_samples_path)
    poly_ddsm = inner_poly_calculation(ddsm_samples_path)
    poly_wgan = inner_poly_calculation(wgan_samples_path)
    print("Epoch {}: [{}, {}, {}]".format(i, poly_mdm, poly_ddsm, poly_wgan))
    
    data1.append(poly_mdm)
    data2.append(poly_ddsm)
    data3.append(poly_wgan)

plot_poly(data1, data2, data3, "./results/Supp_poly.png")






