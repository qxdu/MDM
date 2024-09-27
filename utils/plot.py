import os, sys
from Bio import motifs
from Bio.Seq import Seq
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

from gpro.utils.base import open_fa
import random
import Levenshtein

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import entropy

from scipy.stats import pearsonr
from gpro.evaluator.kmer import convert, kmer_count, get_kmer_stat


font_label = {'family' : 'Arial', 'weight' : 'normal', 'size'   : 10}

def find_locate(arnt,file,res_start,res_end,max_read_num,th):
    with open(file) as handle:   #read the pfm matrix
         srf = motifs.read(handle, "pfm")        
    pwm = srf.counts.normalize(pseudocounts=0.5)    
    background = {"T":0.291,"C":0.218,"G":0.225,"A":0.266}
    pssm = pwm.log_odds(background)
    print(pssm)
        
    result = []
    locate = []
    i = 0
    while i < max_read_num:
        tmp = arnt.instances[i][res_start:res_end]
        tmp = pssm.calculate(tmp)
        result.append(tmp)
        if np.max(tmp) > th:
            locate.append(np.argmax(tmp) + res_start)
        else:
            locate.append(-1)
        i = i + 1
    return result,locate

def draw_dif_distribution(locate_10,locate_35,select_color,p):
    distance = []
    i = 0
    while i < len(locate_10):
        if locate_10[i] != -1 and locate_35[i] != -1:
            distance.append(locate_10[i] - locate_35[i] - 6)
        i = i + 1
    distance = np.matrix(distance)
    distance = distance.T
    
    min_num = 8
    max_num = 27
    transparency = 0.8
    
    distance = distance.T
    distance = np.array(distance)
    distance = distance.T
    distance = distance[:,0]
    
    density = gaussian_kde(distance)
    xs = np.linspace(0,30,200)
    density.covariance_factor = lambda : 0.5
    density._compute_covariance()
    
    return density

def input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,draw_color,max_read_num,p,th):
    with open(file) as handle:
        arnt = motifs.read(handle, "sites")    
    
    file = '../Figure2_analysis/-35.pfm'
    result_35,locate_35 = find_locate(arnt,file,res_start_35,res_end_35,max_read_num,th)   
    file = '../Figure2_analysis/-10.pfm'
    result_10,locate_10 = find_locate(arnt,file,res_start_10,res_end_10,max_read_num,th)      
    distance = draw_dif_distribution(locate_10,locate_35,draw_color,p)
    
    return locate_10,locate_35,distance



def inner_levenshtein_distance_calculation(samples_path, N=1000):
    samples = open_fa(samples_path)
    samples = random.sample(samples, N)
    
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

def inner_CG_calculation(samples_path):
    seqs = open_fa(samples_path)
    total_counts = []
    for i in range(len(seqs)):
        total_counts.append(seqs[i].count("C") + seqs[i].count("G"))
    res = sum(total_counts) / len(total_counts)    
    return res

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
    return poly_total_res[0]


def plot_diversity(data1, data2, data3, report_path):
    data_len = len(data1)
    ylabel = "Inner Distance"
    file_tag = "inner_levenshtein_distance"
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (12,4), dpi = 600)
    x = list(range(0, data_len))

    palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']
    plt.plot(x, data1, '^-', label = 'mdm', linewidth = 0.7,markersize=5, markeredgecolor = 'black', color = 'tab:orange', alpha = 0.8)
    plt.plot(x, data1, '-', color = 'tab:orange', linewidth = 0.7)
    plt.plot(x, data2, 'o-', label = 'ddsm', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
    plt.plot(x, data2, '-', color = 'tab:blue', linewidth = 0.7)
    plt.plot(x, data3, 'x-', label = 'wgan', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:green', alpha = 0.8)
    plt.plot(x, data3, '-', color = 'tab:green', linewidth = 0.7)
    
    # plt.legend(loc="lower left", markerscale = 1) # upper
    plt.legend(loc="lower left", bbox_to_anchor=(0.3, 1), ncol=3, markerscale = 1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(0, data_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, data_len, 10)], fontsize = 12)

    plotnamefinal = report_path
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    print('Results saved to ' + plotnamefinal)

def plot_cg_content(data1, data2, data3, report_path):
    data_len = len(data1)
    ylabel = "CG Counts"
    file_tag = "inner_cytosine_guanine_counts"
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (12,4), dpi = 600)
    x = list(range(0, data_len))

    palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']
    plt.plot(x, data1, '^-', label = 'mdm', linewidth = 0.7,markersize=5, markeredgecolor = 'black', color = 'tab:orange', alpha = 0.8)
    plt.plot(x, data1, '-', color = 'tab:orange', linewidth = 0.7)
    plt.plot(x, data2, 'o-', label = 'ddsm', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
    plt.plot(x, data2, '-', color = 'tab:blue', linewidth = 0.7)
    plt.plot(x, data3, 'x-', label = 'wgan', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:green', alpha = 0.8)
    plt.plot(x, data3, '-', color = 'tab:green', linewidth = 0.7)
    
    # plt.legend(loc="lower left", markerscale = 1) # upper
    plt.legend(loc="lower left", bbox_to_anchor=(0.3, 1), ncol=3, markerscale = 1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(0, data_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, data_len, 10)], fontsize = 12)

    plotnamefinal = report_path
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    print('Results saved to ' + plotnamefinal)
    
def plot_poly(data1, data2, data3, report_path):
    data_len = len(data1)
    ylabel = "Poly AT Counts"
    file_tag = "poly_adenine_thymine_counts"
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (12,4), dpi = 600)
    x = list(range(0, data_len))

    palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']
    plt.plot(x, data1, '^-', label = 'mdm', linewidth = 0.7,markersize=5, markeredgecolor = 'black', color = 'tab:orange', alpha = 0.8)
    plt.plot(x, data1, '-', color = 'tab:orange', linewidth = 0.7)
    plt.plot(x, data2, 'o-', label = 'ddsm', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
    plt.plot(x, data2, '-', color = 'tab:blue', linewidth = 0.7)
    plt.plot(x, data3, 'x-', label = 'wgan', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:green', alpha = 0.8)
    plt.plot(x, data3, '-', color = 'tab:green', linewidth = 0.7)
    
    # plt.legend(loc="lower left", markerscale = 1) # upper
    plt.legend(loc="lower left", bbox_to_anchor=(0.3, 1), ncol=3, markerscale = 1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(0, data_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, data_len, 10)], fontsize = 12)

    plotnamefinal = report_path
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    print('Results saved to ' + plotnamefinal)


def inner_kmer_frequency_calculation(samples_path, natural_path, K=6):
    samples = open_fa(samples_path)
    natural = open_fa(natural_path)
    
    kmer_stat_control, kmer_stat_model = get_kmer_stat(samples, natural, K)
    
    kmer_control_list = list(kmer_stat_control.items())
    kmer_model_list = list(kmer_stat_model.items())
    control_mer, model_mer = [], []
    control_val, model_val = [], []
    ratio = []

    for i in range( pow(4,K) ):
        control_mer.append(kmer_control_list[i][0].upper())
        control_val.append(kmer_control_list[i][1])
        model_mer.append(kmer_model_list[i][0])
        model_val.append(kmer_model_list[i][1])
        if control_val[i] != 0:
            ratio.append(model_val[i]/control_val[i])
    pearsonr_val = pearsonr(model_val, control_val)[0]
    return pearsonr_val

def plot_kmer_frequency(data1, data2, data3, report_path):
    data_len = len(data1)
    ylabel = "6-mer Motif Frequency"
    file_tag = "inner_kmer_frequency"
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (12,4), dpi = 600)
    x = list(range(0, data_len))

    palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']
    plt.plot(x, data1, '^-', label = 'mdm', linewidth = 0.7,markersize=5, markeredgecolor = 'black', color = 'tab:orange', alpha = 0.8)
    plt.plot(x, data1, '-', color = 'tab:orange', linewidth = 0.7)
    plt.plot(x, data2, 'o-', label = 'ddsm', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
    plt.plot(x, data2, '-', color = 'tab:blue', linewidth = 0.7)
    plt.plot(x, data3, 'x-', label = 'wgan', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:green', alpha = 0.8)
    plt.plot(x, data3, '-', color = 'tab:green', linewidth = 0.7)

    plt.legend(loc="lower left", bbox_to_anchor=(0.3, 1), ncol=3, markerscale = 1)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(0, data_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, data_len, 10)], fontsize = 12)

    plotnamefinal = report_path
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    plt.close()
    
    print('Results saved to ' + plotnamefinal)


'''
DECIPHERING
'''


def cal_max_pos(seq,PPM_temp,tag="start"):
    res_dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    all_values = np.zeros(len(seq)+1-len(PPM_temp))
    for i in range(len(seq)+ 1 - len(PPM_temp)):
        value = 1
        for k in range(len(PPM_temp)):
            value *= PPM_temp[k][res_dic[seq[i + k]]]
        all_values[i] = value
    all_max_pos = np.argwhere(all_values==np.max(all_values))
    
    if tag=="start":
        max_pos = all_max_pos[0]
    elif tag=="end":
        max_pos = all_max_pos[-1]
    return max_pos, np.max(all_values)

def cal_distance(file, ppm1, ppm2):
    seqs = open_fa(file)
    distance = []
    for seq in seqs:
        start, flag1 = cal_max_pos(seq, ppm1, "start")
        end, flag2 = cal_max_pos(seq, ppm2, "end")
        gap = end - start - len(ppm1)
        if flag1!=0 and flag2!=0 and gap>0:
            distance.append(int(gap))
    
    try:
        density = gaussian_kde(distance)
        xs = np.linspace(0,30,200)
        density.covariance_factor = lambda : 0.25
        density._compute_covariance()
    except:
        return "error", "error"
    
    return distance, density

def open_pfm(file):
    f = open(file,'r')
    lines = f.readlines()
    lines = lines[1:]
    
    rows = len(lines)
    cols = len(lines[0].strip('\n').split())
    res = np.zeros((rows, cols), dtype=float)
    
    for i in range(rows):
        cur_row = lines[i].strip('\n').split()
        for j in range(cols):
            res[i][j] = float( cur_row[j] )
    
    res = pfm_to_ppm(res)
    
    return res
    

def pfm_to_ppm(pfm):
    motif_len = len(pfm[0])
    res = np.zeros((motif_len, 4))
    
    for i in range(motif_len):
        cnt = sum(pfm[:,i])
        for j in range(4):
            res[i,j] = pfm[j,i]/cnt
    return res
    
def plot_boxplot(plot_path, barplot_data):
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

    ax = sns.boxplot( x="labels", y="distance",  data=barplot_data,  boxprops=dict(alpha=.9), # hue="label", hue_order = hue_order,
                      fliersize=1, flierprops={"marker": 'x'}, hue_order=["mdm", "ddsm", "wgan"], palette="deep") # # palette="viridis_r"
    h,_ = ax.get_legend_handles_labels()

    ax.set_xlabel('Models', fontsize=10)
    ax.set_ylabel('Diversity', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.show()
    plt.savefig(plot_path)
