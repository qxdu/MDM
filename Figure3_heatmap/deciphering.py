
'''
part5: deciphering
'''

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import time
import tqdm
import pandas as pd
from matplotlib import pyplot as plt

import sys
sys.path.append("../utils/")


from mds import *
from plot import *

import logging
logger = logging.getLogger()



'''
decoupling
'''

origin = './samples/falsify.txt'
ddsm = "./samples/ddsm_35000.txt"
mdm = './samples/mdm_35000.txt'
wgan = "./samples/wgan_35000.txt"


pfm_0266 = "./library/MA0266.1.pfm"
pfm_0267 = "./library/MA0267.1.pfm"
pfm_0277 = "./library/MA0277.1.pfm"
pfm_0280 = "./library/MA0280.1.pfm"
pfm_0300 = "./library/MA0300.1.pfm"
pfm_0306 = "./library/MA0306.1.pfm"

pfm_list = [266, 267, 277, 280, 300, 306]

res_natural, res_ddsm, res_mdm, res_wgan = [], [], [], []
item_list = []

cnt = 0
for item in tqdm(itertools.product(pfm_list, pfm_list)):
    PFM1 = "./library/MA0{}.1.pfm".format(str(item[0]))
    PFM2 = "./library/MA0{}.1.pfm".format(str(item[1]))
    item_list.append(item)
    
    ppm1 = open_pfm(PFM1)
    ppm2 = open_pfm(PFM2)
    dist_origin, density_origin = cal_distance(origin, ppm1, ppm2)
    dist_ddsm, density_ddsm = cal_distance(ddsm, ppm1, ppm2)
    dist_mdm, density_mdm = cal_distance(mdm, ppm1, ppm2)
    dist_wgan, density_wgan = cal_distance(wgan, ppm1, ppm2)
    
    print(density_wgan)
    
    dist = [dist_origin, dist_ddsm, dist_mdm, dist_wgan]
    
    xs = np.linspace(0,30,200)
    
    res_ddsm.append(entropy(density_origin(xs), density_ddsm(xs)))
    res_mdm.append(entropy(density_origin(xs), density_mdm(xs)))
    res_wgan.append(entropy(density_origin(xs), density_wgan(xs)))

res_ddsm = -np.log2(res_ddsm).reshape(len(pfm_list), len(pfm_list)).T
res_mdm = -np.log2(res_mdm).reshape(len(pfm_list), len(pfm_list)).T
res_wgan = -np.log2(res_wgan).reshape(len(pfm_list), len(pfm_list)).T

def plot_prepare(X):
    x_tick= pfm_list
    y_tick= pfm_list
    data={}
    for i in range(6):
        data[x_tick[i]] = X[i]
    pd_data=pd.DataFrame(data,index=y_tick,columns=x_tick)
    return pd_data

res_ddsm = plot_prepare(res_ddsm)
res_mdm = plot_prepare(res_mdm)
res_wgan = plot_prepare(res_wgan)

res_delta = res_mdm - res_ddsm

font = {'size'   : 8}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
plt.rcParams["figure.dpi"] = 300


# ax = sns.heatmap(data=res_ddsm,annot=False, linewidths=0.3,cmap="RdBu_r", vmin=0, vmax=12, fmt=".1f")  # n_colors = 20
# plotnamefinal = "./results/Fig3_heat_ddsm.png"

# ax = sns.heatmap(data=res_mdm,annot=False, linewidths=0.3,cmap="RdBu_r", vmin=0, vmax=12, fmt=".1f")  # n_colors = 20
# plotnamefinal = "./results/Fig3_heat_mdm.png"

# ax = sns.heatmap(data=res_wgan,annot=False, linewidths=0.3,cmap="RdBu_r", vmin=0, vmax=12, fmt=".1f")  # n_colors = 20
# plotnamefinal = "./results/Fig3_heat_wgan.png"

ax = sns.heatmap(data=res_delta,annot=True, linewidths=0.3,cmap="RdBu_r", vmin=-3, vmax=3, fmt=".1f")  # n_colors = 20
plotnamefinal = "./results/Fig3_delta_ddsm_num.png"

plt.savefig(plotnamefinal)
plt.show()
