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

def plot_kmer(generator_training_datapath, generator_sampling_datapath, report_path, file_tag, K=6, num_seqs_to_test=10000):
    training_text = open_fa(generator_training_datapath)
    sampling_text = open_fa(generator_sampling_datapath)
    
    if num_seqs_to_test > min(len(training_text), len(sampling_text)):
        num_seqs_to_test = min(len(training_text), len(sampling_text))

    random.shuffle(training_text)
    random.shuffle(sampling_text)
    training_text = training_text[0:num_seqs_to_test]
    sampling_text = sampling_text[0:num_seqs_to_test]
    
    ## k-mer estimation
    kmer_stat_control, kmer_stat_model = get_kmer_stat(sampling_text, training_text, K)
    kmer_control_list = list(kmer_stat_control.items())
    kmer_model_list   = list(kmer_stat_model.items())
    control_mer, model_mer, control_val, model_val, ratio = [], [], [], [], []

    for i in range( pow(4,K) ):
        control_mer.append(kmer_control_list[i][0].upper())
        control_val.append(kmer_control_list[i][1])
        model_mer.append(kmer_model_list[i][0])
        model_val.append(kmer_model_list[i][1])
        if control_val[i] != 0:
            ratio.append(model_val[i]/control_val[i])
    pearsonr_val = pearsonr(model_val, control_val)[0]
    boundary = max(max(control_val), max(model_val))
    
    print("Model Pearson Correlation: {}, Boundary: {}".format(pearsonr_val, boundary) )
    
    bound = 0.0016
    
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,5), dpi = 300) # (6,5)
    
    plt.xlim(0,bound)
    plt.ylim(0,bound)
    
    # color_for_plot = "RoyalBlue" # mdm, 0.00148
    # color_for_plot = "mediumpurple" # ddsm, 0.0013
    # color_for_plot = "IndianRed" # wgan, 0.004
    # color_for_plot = "slategray" # pssm
    color_for_plot = "hotpink" # promodiff

    sns.regplot(x=control_val, y=model_val, color=color_for_plot, scatter=False, truncate=False)
    plt.scatter(control_val,model_val, c=color_for_plot, label=round(pearsonr_val,3), marker=".", s=30, alpha = 0.8, linewidths=0) # , marker=".", s=6, alpha = 0.8
    ax.set_xlabel("Natural Promoters", fontsize=12)
    ax.set_ylabel("Generated Promoters", fontsize=12)
    
    x = np.linspace(0,bound,100)
    y = np.linspace(0,bound,100)
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.linspace(0, bound, 5))
    ax.set_yticks(np.linspace(0, bound, 5))
    
    plt.annotate('', xy=(1.08, 0), xycoords='axes fraction',
             xytext=(-0.06, 0), textcoords='axes fraction',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             zorder=5)
    
    plt.annotate('', xy=(0, 1.10), xycoords='axes fraction',
             xytext=(0, -0.06), textcoords='axes fraction',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             zorder=5)
    
    plotnamefinal = report_path + 'kmer_' + file_tag + ".png"
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    
    print('Kmer frequency plot saved to ' + plotnamefinal)
    return (pearsonr_val, plotnamefinal)

generator_training_datapath = "/home/qxdu/AI_based_promoter_design/dataset/diffusion_promoter/sequence_data.txt"
# generator_sampling_datapath = "/home/qxdu/AI_based_promoter_design/samples/selection/diff_rand_10000.txt" # mdm
# generator_sampling_datapath = "/home/qxdu/AI_based_promoter_design/revisions/checkpoints/samples/ddsm_epoch_200.txt" # ddsm
# generator_sampling_datapath = "/home/qxdu/AI_based_promoter_design/samples/selection/wgan_rand_10000.txt" # wgan

# generator_sampling_datapath = "/home/qxdu/AI_based_promoter_design/samples/selection/ppm_rand_10000.txt" # pssm
generator_sampling_datapath = "/home/qxdu/AI_based_promoter_design/revisions/checkpoints/promodiff/output_genepromodiff.txt" # promodiff

plot_kmer(generator_training_datapath, generator_sampling_datapath, report_path="./figs/", file_tag="promodiff")




'''
mdm: Model Pearson Correlation: 0.928696114468897, Boundary: 0.00148
ddsm: Model Pearson Correlation: 0.9269882615121661, Boundary: 0.0014
wgan: Model Pearson Correlation: 0.68448200009648, Boundary: 0.004

pssm: Model Pearson Correlation: 0.6101778369005526, Boundary: 0.0012
promodiff: Model Pearson Correlation: 0.4110841850829727, Boundary: 0.0013
'''