'''
放置data_driven的3个metrics, 效仿deepcross的方法进行绘图
1. GC content, JS distance
2. Kmer similarity
3. Diversity

similarity: 为MDM, DDSM, WGAN, PSSM 在4个数据集上各生成10000个样本
diversity: 为MDM, DDSM, WGAN, PSSM 在4个数据集上各生成100个随机数种子下的各100个样本

'''
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy
import Levenshtein
from itertools import product
from scipy.stats import pearsonr


from gpro.utils.base import write_fa, write_seq, write_exp
from gpro.evaluator.kmer import convert, kmer_count, get_kmer_stat
from gpro.utils.utils_predictor import EarlyStopping, seq2onehot, open_fa, open_exp


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages


def get_kmer_pcc(sampling_path, training_path, K=6):
    sampling_text = open_fa(sampling_path)
    training_text = open_fa(training_path)
    
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
    return pearsonr_val

def seq_GC(seq):
    GCcount=0
    for i in range(len(seq)): 
        if(seq[i])=='G'or (seq[i])=='C':
            GCcount+=1
    return GCcount/len(seq)*100

def get_GC_content(fasta_file):
    with open(fasta_file, 'r') as f:
        GC=[]
        for line in f.readlines():
            if not line.startswith(">"):
                seq = line.strip()
                GC.append(seq_GC(seq))
    return GC 

def JS_divergence(p,q):
    p, q = np.array(p), np.array(q)
    js_list = []
    n = min(len(p), len(q))
    
    for k in range(3):
        if len(p) > len(q): # sample > natural
            seq_index_A = np.arange(q.shape[0])
            np.random.seed(k+2)
            np.random.shuffle(seq_index_A)
            p = p[seq_index_A[:n]]
        else:               # sample < natural
            seq_index_A = np.arange(p.shape[0])
            np.random.seed(k+2)
            np.random.shuffle(seq_index_A)
            q = q[seq_index_A[:n]]
        M=(p+q)/2
        each_js = 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)
        js_list.append(each_js)
    return np.mean(js_list)

check_list = [ "ecoli_50", "ecoli_165", "yeast_80", "yeast_1000"]
model_list = ["ddsm", "mdm", "wgan", "pssm"]

tag_list = [ "ecoli_50_Wanglab", "ecoli_165bp_Wanglab", "yeast_80bp_Avivlab", "yeast_1000bp_ZelezniakLab"] 
natural_list = [tag for tag in tag_list for _ in range(4)]

prefix_list = list(product(check_list, model_list))
js_list, kmer_list = [], []
check_list, model_list = [], []

## similarity

for i, (check, model) in enumerate(prefix_list):
    sample_path = "../samples/{}_{}.txt".format(check, model)
    natural_path = "../dataset/{}/train_seq.txt".format(natural_list[i])
    
    sample_gc, natural_gc = get_GC_content(sample_path), get_GC_content(natural_path)
    
    js_val = abs( np.mean(sample_gc)- np.mean(natural_gc) ) / np.mean(natural_gc)
    js_list.append(js_val)
    
    # js_val = JS_divergence(sample_gc, natural_gc) 
    # js_list.append(js_val)

    kmer_val = get_kmer_pcc(sample_path, natural_path, K=6)
    kmer_list.append(kmer_val)
    
    check_list.append(check)
    model_list.append(model)
    
    print(i, check, model, round(js_val,4), round(kmer_val, 4) )


## plot similarity

color_list = ["#F9A319", "#4D68B0", "#891619", "slategray"]
# color_list = ["royalblue", "mediumpurple", "indianred", "slategray"]

def plot_kmer(plot_path, barplot_data):
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 600)

    ax = sns.barplot(x="benchmark", y="kmer_val", hue="model", data=barplot_data, palette=color_list, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    
    ax.set_xlabel("")
    ax.set_ylabel("PCC of 6-mer frequency with natural benchmarks")

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.08, 1), ncol=4)

    plt.title("")
    plt.show()
    plt.savefig(plot_path)

def plot_gc(plot_path, barplot_data):
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 600)

    ax = sns.barplot(x="benchmark", y="js_val", hue="model", data=barplot_data, palette=color_list, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    
    ax.set_xlabel("")
    ax.set_ylabel("JS divergence to natural GC content distribution")

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.08, 1), ncol=4)

    plt.title("")
    plt.show()
    plt.savefig(plot_path)


df_kmer = pd.DataFrame({"benchmark": check_list, "model": model_list, "kmer_val": kmer_list})
df_js = pd.DataFrame({"benchmark": check_list, "model": model_list, "js_val": js_list})

plot_kmer("./results/Fig2_kmer.png", df_kmer)
plot_gc("./results/Fig2_GC.png", df_js)

