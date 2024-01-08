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
sys.path.append("./utils/")

from mds import *
from plot import *

from scipy import stats

max_read_num = 1000
res_start_35 = 10
res_end_35 = 25
res_start_10 = 35
res_end_10 = 50
draw_color = 'blue'
th = -1000

file = '/home/qxdu/AI_based_promoter_design/dataset/diffusion_promoter/sequence_data.fa'
locate_10_1,locate_35_1,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'green',max_read_num,1,th)

file =  "/home/qxdu/AI_based_promoter_design/samples/selection/diff_rand_10000.txt"
locate_10_2,locate_35_2,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'orange',max_read_num,1,th)

file = "/home/qxdu/AI_based_promoter_design/revisions/checkpoints/samples/ddsm_epoch_200.txt"
locate_10_3,locate_35_3,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'blue',max_read_num,1,th)

file = "/home/qxdu/AI_based_promoter_design/revisions/checkpoints/promodiff/output_genepromodiff.txt"
locate_10_4,locate_35_4,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'red',max_read_num,1,th)

file = "/home/qxdu/AI_based_promoter_design/samples/selection/wgan_rand_10000.txt"
locate_10_5,locate_35_5,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'purple',max_read_num,1,th)

file = "/home/qxdu/AI_based_promoter_design/evaluation/gap_simulation/core_motif_simulation/pwm_seq.sites"
locate_10_6,locate_35_6,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'yellow',max_read_num,1,th)

density1 = draw_dif_distribution(locate_10_1,locate_35_1,draw_color,2)
density2 = draw_dif_distribution(locate_10_2,locate_35_2,draw_color,3)
density3 = draw_dif_distribution(locate_10_3,locate_35_3,draw_color,4)
density4 = draw_dif_distribution(locate_10_4,locate_35_4,draw_color,5)
density5 = draw_dif_distribution(locate_10_5,locate_35_5,draw_color,6)
density6 = draw_dif_distribution(locate_10_6,locate_35_6,draw_color,7)


fig, ax = plt.subplots()

plt.tick_params(labelsize=14,width = 2, length = 5)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


xs = np.linspace(0,30,200)
plt.plot(xs,density1(xs),color='green', label="Natural")
plt.plot(xs,density2(xs),color='orange', label="MDM")
plt.plot(xs,density3(xs),color='darkred', label="DDSM")
plt.plot(xs,density4(xs),color='royalblue', label="PromoDiff")
plt.plot(xs,density5(xs),color='blueviolet', label="WGAN")
plt.plot(xs,density6(xs),color='slategray', label="PSSM")


plt.xlabel('distance',font_label)
plt.ylabel('count',font_label)
plt.legend(loc='upper left')
plt.savefig("./figs/gap_all.png")
plt.show()