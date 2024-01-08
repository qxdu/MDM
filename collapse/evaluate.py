import Levenshtein
import sys
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import random
from scipy.stats import pearsonr

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from gpro.utils.base import open_fa
from gpro.evaluator.kmer import convert, kmer_count, get_kmer_stat


natural_path = "./dataset/sequence_data.txt"
wgan_sample_dir = "./checkpoints/wgan/samples"
diff_sample_dir = "./checkpoints/diffusion/samples"

##########################
### Calculate Distance ###
##########################

        
def inner_levenshtein_distance_calculation(samples_path):
    samples = open_fa(samples_path)
    samples_length = len(samples)
    
    res_list = []
    for i in range(samples_length):
        seq_dist = []
        for j in range(samples_length):
            if j!=i:
                dis = Levenshtein.distance(samples[i],samples[j])
                seq_dist.append(dis)
        res_list.append(min(seq_dist))
    avg_dis = np.sum(res_list)/len(res_list)
    return avg_dis


def transitional_levenshtein_distance_calculation(samples_path1, samples_path2):
    samples1 = open_fa(samples_path1)
    samples2 = open_fa(samples_path2)
    samples_length = len(samples1)
    
    res_list = []
    for i in range(samples_length):
        seq_dist = []
        for j in range(samples_length):
            dis = Levenshtein.distance(samples1[i],samples2[j])
            seq_dist.append(dis)
        res_list.append(min(seq_dist))
    avg_dis = np.sum(res_list)/len(res_list)
    return avg_dis


def inner_kmer_frequency_calculation(samples_path, natural_path):
    samples = open_fa(samples_path)
    natural = open_fa(natural_path)
    # natural = random.sample(natural, 100)
    
    kmer_stat_control, kmer_stat_model = get_kmer_stat(samples, natural, 6)
    
    kmer_control_list = list(kmer_stat_control.items())
    kmer_model_list = list(kmer_stat_model.items())
    control_mer, model_mer = [], []
    control_val, model_val = [], []
    ratio = []

    for i in range( pow(4,6) ):
        control_mer.append(kmer_control_list[i][0].upper())
        control_val.append(kmer_control_list[i][1])
        model_mer.append(kmer_model_list[i][0])
        model_val.append(kmer_model_list[i][1])
        if control_val[i] != 0:
            ratio.append(model_val[i]/control_val[i])
    pearsonr_val = pearsonr(model_val, control_val)[0]
    return pearsonr_val


def inner_poly_calculation(samples_path):
    
    seqs = open_fa(samples_path)
    polyA_res = []
    polyT_res = []
    poly_total_res = []
    scores = 0
    
    for K in range(4,9):
        motif_A = "A" * K
        motif_T = "T" * K

        total_counts = []

        for i in range(len(seqs)):
            total_counts.append(seqs[i].count(motif_A) + seqs[i].count(motif_T))
        poly_total_res.append(np.mean(total_counts))
    return poly_total_res


def inner_CG_calculation(samples_path):
    seqs = open_fa(samples_path)
    total_counts = []
    for i in range(len(seqs)):
        total_counts.append(seqs[i].count("C") + seqs[i].count("G"))
    res = sum(total_counts) / len(total_counts)    
    return res


data1 = []
data2 = []
mode = "cg"


for i in range(1, 60+1):
    wgan_samples_path = wgan_sample_dir + "/sample_ep{}_s0_num_100.txt".format(i)
    diff_samples_path = diff_sample_dir + "/sample_ep{}_s0_num_100.txt".format(i)
    
    ## 1. Inner Distance
    if mode == "inner":
        avg_dis_wgan = inner_levenshtein_distance_calculation(wgan_samples_path)
        avg_dis_diff = inner_levenshtein_distance_calculation(diff_samples_path)
        # print("Epoch {}: [{}, {}]".format(i, avg_dis_wgan, avg_dis_diff))
        data1.append(avg_dis_wgan)
        data2.append(avg_dis_diff)
    
    
    ## 2. Transitional Distance
    if mode == "transitional":
        wgan_samples_path_next = wgan_sample_dir + "/sample_ep{}_s0_num_100.txt".format(i+1)
        diff_samples_path_next = diff_sample_dir + "/sample_ep{}_s0_num_100.txt".format(i+1)

        avg_dis_wgan = transitional_levenshtein_distance_calculation(wgan_samples_path, wgan_samples_path_next)
        avg_dis_diff = transitional_levenshtein_distance_calculation(diff_samples_path, diff_samples_path_next)
        # print("Epoch {} -> {}: [{}, {}]".format(i, i+1, avg_dis_wgan, avg_dis_diff))
        data1.append(avg_dis_wgan)
        data2.append(avg_dis_diff)
    
    ## 3. poly A,T
    if mode == "poly":
        avg_poly_wgan = inner_poly_calculation(wgan_samples_path)
        avg_poly_diff = inner_poly_calculation(diff_samples_path)
        # print("Epoch {}: [{}, {}]".format(i, avg_poly_wgan, avg_poly_diff))
        data1.append(avg_poly_wgan[0])
        data2.append(avg_poly_diff[0])
    
    ## 4. kmer frequency (for 10000 samples)
    if mode == "kmer":
        natural_path = "./dataset/sequence_data.txt"
        kmer_wgan = inner_kmer_frequency_calculation(wgan_samples_path, natural_path)
        kmer_diff = inner_kmer_frequency_calculation(diff_samples_path, natural_path)
        data1.append(kmer_wgan)
        data2.append(kmer_diff)
        
    ## 5. CG content
    if mode == "cg":
        cg_wgan = inner_CG_calculation(wgan_samples_path)
        cg_diff = inner_CG_calculation(diff_samples_path)
        data1.append(cg_wgan)
        data2.append(cg_diff)
    
## 5. plot

if mode == "inner":
    ylabel = "Inner Distance"
    file_tag = "inner_levenshtein_distance"
elif mode == "transitional":
    ylabel = "Transtional Distance"
    file_tag = "transitional_levenshtein_distance"
elif mode == "poly":
    ylabel = "Poly AT Counts"
    file_tag = "poly_adenine_thymine_counts"
elif mode == "kmer":
    ylabel = "6-mer Motif Frequency"
    file_tag = "inner_kmer_frequency"
elif mode == "cg":
    ylabel = "CG Counts"
    file_tag = "inner_cytosine_guanine_counts"


data_len = len(data1)
report_path = "./plot/"


font = {'size' : 12}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
x = list(range(0, data_len))

palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']
plt.plot(x, data1, '^-', color = 'tab:orange', linewidth = 0.7, label = 'wgan',markeredgecolor = 'black', markersize=5, alpha = 0.8)
plt.plot(x, data1, '-', color = 'tab:orange', linewidth = 0.7)
plt.plot(x, data2, 'o-', label = 'diffusion', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
plt.plot(x, data2, '-', color = 'tab:blue', linewidth = 0.7)
plt.legend(loc="lower left", markerscale = 1) # upper

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)
plt.tick_params(length = 10)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xticks(np.arange(0, data_len, 10))
ax.set_xticklabels([str(x) for x in np.arange(0, data_len, 10)], fontsize = 12)

plotnamefinal = report_path + file_tag + ".png"
plt.tight_layout()
plt.savefig(plotnamefinal)
print('Results saved to ' + plotnamefinal)


    
    

##########################
###  Ploting Distance  ###
##########################


def data_rearrangement(distance_list):
    res = np.array(distance_list)
    idx = sorted(range(len(res)), key=lambda k: res[k], reverse=True)
    res = res[idx]
    return res

## Ploting figures: Min
# wgan_min_data = data_rearrangement(wgan_min_distance)
# wgan_max_data = data_rearrangement(wgan_max_distance)
# diff_min_data = data_rearrangement(diff_min_distance)
# diff_max_data = data_rearrangement(diff_max_distance)
# # ppm_min_data = data_rearrangement(ppm_min_distance)
# # ppm_max_data = data_rearrangement(ppm_max_distance)

# plt.plot(wgan_min_data, color="#778899")
# plt.plot(diff_min_data, color="#9370DB")
# # plt.plot(ppm_min_data, color="#4169E1")
# plt.legend( ['wgan','diffusion'] )

# plt.xlabel("distance")
# plt.title("Minimum Levenshtein Distance Distribution")
# plt.show()
# plt.savefig("ls_dist_min.png")
# plt.close()


# ## Ploting figures: Max
# plt.plot(wgan_max_data, color="#778899")
# plt.plot(diff_max_data, color="#9370DB")
# # plt.plot(ppm_max_data, color="#4169E1")
# plt.legend( ['wgan','diffusion'] )

# plt.xlabel("distance")
# plt.title("Maximum Levenshtein Distance Distribution")
# plt.show()
# plt.savefig("ls_dist_max.png")
# plt.close()