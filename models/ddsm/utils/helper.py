import os
import sys
import functools
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

import torch
from torch import nn
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append(".../ddsm-main/")
sys.path.append(".../ddsm-main/external/")

from ddsm import *
from sei import *

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

class SequenceData(Dataset):
  def __init__(self,data, label):
    self.data = data
    self.target = label
  
  def __getitem__(self, index):
    return self.data[index], self.target[index]
    
  def __len__(self):
    return self.data.size(0)
  
  def __getdata__(self):
    return self.data, self.target

class TestData(Dataset):
    def __init__(self,data):
        self.data = data
  
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return self.data.size(0)
    
    def __getdata__(self):
        return self.data

def seq2onehot(seq,length):
        module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        i = 0
        promoter_onehot = []
        while i < len(seq):
           tmp = []
           for item in seq[i]:
                if item == 'a' or item == 'A':
                    tmp.append(module[0])
                elif item == 'c' or item == 'C':
                    tmp.append(module[1])
                elif item == 'g' or item == 'G':
                    tmp.append(module[2])
                elif item == 't' or item == 'T':
                    tmp.append(module[3])
                else:
                    tmp.append([0,0,0,0])
           promoter_onehot.append(tmp)
           i = i + 1
        data = np.zeros((len(seq),length,4))
        data = np.float32(data)
        i = 0
        while i < len(seq):
            j = 0
            while j < len(seq[0]):
                data[i,j,:] = promoter_onehot[i][j]
                j = j + 1
            i = i + 1
        return data

def onehot2seq(onehot, seq_len = 50):
    onehot = onehot.view(-1, seq_len, 4).cpu().detach().numpy()
    ref = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_list = []
    for item in onehot:
        seq = ''
        for letter in item:
            letter = int(np.where(letter == np.amax(letter))[0])
            seq = seq + ref[letter]
        if seq != '':
            seq_list.append(seq)
    return seq_list

def write_fa(file,data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write('>' + str(i) + '\n')
        f.write(data[i] + '\n')
        i = i + 1
    f.close()


def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record


#-------------------------------------------------------------------------------------------#

class Dense(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)

class ScoreNet(nn.Module):

    def __init__(self, embed_dim=256, time_dependent_weights=None, time_step=0.01):
        
        super().__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        n = 256
        self.linear = nn.Conv1d(4, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step

    def forward(self, x, t, t_ind=None, return_a=False):

        embed = self.act(self.embed(t / 2))
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)
        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out

