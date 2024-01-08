import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from tqdm import tqdm

from gpro.utils.utils_predictor import open_exp
from gpro.utils.base import write_fa, write_seq
from gpro.predictor.cnn_k15.cnn_k15 import CNN_K15_language
from gpro.evaluator.kmer import convert, kmer_count, get_kmer_stat

from scipy.stats import pearsonr

def random_sequence_generation(sample_len, sample_num):
    
    nt_map = ["A", "C", "G", "T"]
    cnt = 0
    
    substrate = []
    while cnt < sample_num:
        sample = []
        for i in range(sample_len):
            nt = np.random.choice(nt_map)
            sample.append(nt)
        substrate.append( "".join(sample) )
        cnt = cnt + 1
    
    return substrate

def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def write_fa(file,data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write('>' + str(i) + '\n')
        f.write(data[i] + '\n')
        i = i + 1
    f.close()
    
def get_ref_motifs(datapath):
    inputs = pd.read_csv(datapath,sep='\t', header=0)
    inputs = inputs[~pd.isna(inputs["sequence_name"])]
    
    tmp = list(inputs["motif_id"])
    motif_name_list = list(set(tmp))
    motif_name_list.sort(key=tmp.index)
    
    return motif_name_list
    

def fimo_to_matrix(datapath, savepath, refmotifs):
    inputs = pd.read_csv(datapath,sep='\t', header=0)
    inputs = inputs[~pd.isna(inputs["sequence_name"])]

    tmp = list(inputs["sequence_name"])
    sequence_name_list = list(set(tmp))
    sequence_name_list.sort(key=tmp.index)
    sequence_name_length = len(sequence_name_list) # 50

    motif_name_list = refmotifs
    motif_name_length = len(motif_name_list)

    counter = np.zeros([sequence_name_length, motif_name_length])
    table = pd.DataFrame(counter, columns=motif_name_list, index=sequence_name_list)

    for index, row in tqdm(inputs.iterrows()):
        id = row["motif_id"]
        seq = row["sequence_name"]
        table.loc[seq, id] += 1

    table.to_csv(savepath)

def plot_mds(df_concat, N, savepath):
    df_high = df_concat[0:N]
    df_down = df_concat[-N:]

    sns.set_style("darkgrid")

    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (8,6), dpi = 400)

    sns.scatterplot(x="coordinates.1", y="coordinates.2", data=df_concat, hue = "trend", palette="RdYlGn_r", alpha=1) # palette="viridis_r"
    for _, point in df_high.iterrows():
        ax.text(point["coordinates.1"], point["coordinates.2"], point["labels"], fontsize=8, ha='center', va='bottom')
    for _, point in df_down.iterrows():
        ax.text(point["coordinates.1"], point["coordinates.2"], point["labels"], fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('MDS-1', fontsize=12)
    ax.set_ylabel('MDS-2', fontsize=12)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("")
    plt.savefig(savepath)

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

