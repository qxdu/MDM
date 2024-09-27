
from Bio import motifs
from Bio.Seq import Seq
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import entropy

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

font_label = {
'family' : 'Arial',
'weight' : 'normal',
'size'   : 10,
}


def find_locate(arnt,file,res_start,res_end,max_read_num,th):
    with open(file) as handle:   #read the pfm matrix
         srf = motifs.read(handle, "jaspar")        
    pwm = srf.counts.normalize(pseudocounts=0.5)    
    background = {"T":0.291,"C":0.218,"G":0.225,"A":0.266}
    pssm = pwm.log_odds(background)
    # print(pssm)
        
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
    density.covariance_factor = lambda : 0.15
    density._compute_covariance()
    
    return density


pfm_0266 = "./library/MA0266.1.pfm"
pfm_0267 = "./library/MA0267.1.pfm"
pfm_0277 = "./library/MA0277.1.pfm"
pfm_0280 = "./library/MA0280.1.pfm"
pfm_0300 = "./library/MA0300.1.pfm"
pfm_0306 = "./library/MA0306.1.pfm"


def input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,draw_color,max_read_num,p,th):
    with open(file) as handle:
        arnt = motifs.read(handle, "sites")    
    file = pfm_0280
    result_35,locate_35 = find_locate(arnt,file,res_start_35,res_end_35,max_read_num,th)   
    file = pfm_0300
    result_10,locate_10 = find_locate(arnt,file,res_start_10,res_end_10,max_read_num,th)      
    distance = draw_dif_distribution(locate_10,locate_35,draw_color,p)
    
    return locate_10,locate_35,distance


max_read_num = 5000
res_start_35 = 0
res_end_35 = 35
res_start_10 = 15
res_end_10 = 50
draw_color = 'blue'
th = -1000

file = './samples/falsify.txt'
locate_10_1,locate_35_1,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'green',max_read_num,1,th)

file = './samples/mdm_35000.txt'
locate_10_2,locate_35_2,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'orange',max_read_num,1,th)

file = './samples/wgan_35000.txt'
locate_10_3,locate_35_3,distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'blue',max_read_num,1,th)

file = "./samples/ddsm_35000.txt"
locate_10_4, locate_35_4, distance = input_para(file,res_start_35,res_end_35,res_start_10,res_end_10,'blue',max_read_num,1,th)



density1 = draw_dif_distribution(locate_10_1,locate_35_1,draw_color,2)
density2 = draw_dif_distribution(locate_10_2,locate_35_2,draw_color,3)
density3 = draw_dif_distribution(locate_10_3,locate_35_3,draw_color,4)
density4 = draw_dif_distribution(locate_10_4,locate_35_4,draw_color,4)


fig, ax = plt.subplots()

plt.tick_params(labelsize=10,width = 2, length = 5)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

xs = np.linspace(0,30,200)

print(xs[np.argmax(density1(xs))])
print(xs[np.argmax(density2(xs))])
print(xs[np.argmax(density3(xs))])
print(xs[np.argmax(density4(xs))])

print(-np.log2(entropy(density1(xs), density2(xs))))
print(-np.log2(entropy(density1(xs), density3(xs))))
print(-np.log2(entropy(density1(xs), density4(xs))))


font = {'size' : 10}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
fig, ax = plt.subplots(figsize = (6,6), dpi = 300)

plt.plot(xs,density1(xs),color='dimgrey', label="Natural")
plt.plot(xs,density2(xs),color='green', label="MDM")
plt.plot(xs,density3(xs),color='pink', label="WGAN")
plt.plot(xs,density4(xs),color='royalblue', label="DDSM")

ax.set_xlabel("distance")
ax.set_ylabel("density")

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.1), ncol=4)
plt.tight_layout()
plt.savefig("./results/gap_ddsm_0280_0300.png")
plt.show()














