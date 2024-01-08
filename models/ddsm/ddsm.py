'''
ddsm-main is available at https://github.com/jzhoulab/ddsm/tree/main/promoter_design
gpro is available at https://github.com/WangLabTHU/GPro/tree/main/gpro
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
sys.path.append("./utils/")
sys.path.append("../ddsm-main/")
sys.path.append("../ddsm-main/external/")

from ddsm import *
from sei import *
from helper import *

import logging
logger = logging.getLogger()


diffusion_weights_file = "../ddsm-main/promoter_design/steps400.cat4.speed_balance.time4.0.samples100000.pth"
ncat = 4
n_time_steps = 400
num_epochs = 200
lr = 5e-4
device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
random_order = False
speed_balanced = True

sb = UnitStickBreakingTransform()

v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(diffusion_weights_file)
v_one = v_one.cpu() # torch.Size([100000, 400, 3])
v_zero = v_zero.cpu() # torch.Size([100000, 400, 3])
v_one_loggrad = v_one_loggrad.cpu() # torch.Size([100000, 400, 3])
v_zero_loggrad = v_zero_loggrad.cpu() # torch.Size([100000, 400, 3])
timepoints = timepoints.cpu() # torch.Size([400])
alpha = torch.ones(ncat - 1).float()
beta =  torch.arange(ncat - 1, 0, -1).float()

torch.set_default_dtype(torch.float32)

total_feature = open_fa("/home/qxdu/AI_based_promoter_design/dataset/diffusion_promoter/sequence_data.fa")
total_feature = seq2onehot(total_feature, 50) # ACGT
total_feature = torch.tensor(total_feature, dtype=float) # (sample num,length,4) 
r = int(total_feature.shape[0] * 0.9)

train_set = TestData(total_feature[0:r])
train_dataloader = DataLoader(dataset=train_set, batch_size = 128, shuffle=True)
valid_set = TestData(total_feature[r:])
valid_dataloader = DataLoader(dataset=valid_set, batch_size = 128, shuffle=False)

time_dependent_cums = torch.zeros(n_time_steps).to(device) # 400
time_dependent_counts = torch.zeros(n_time_steps).to(device) # 400

avg_loss = 0.
num_items = 0

for i, x in enumerate(train_dataloader):
    random_t = torch.randint(0, n_time_steps, (x.shape[0],)) # 128 timepoint     
    perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)
    
    perturbed_x = perturbed_x.to(device) # torch.Size([128, 50, 4])
    perturbed_x_grad = perturbed_x_grad.to(device) # torch.Size([128, 50, 4])
    random_t = random_t.to(device)
    perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach() # torch.Size([128, 50, 3])
    time_dependent_counts[random_t] += 1
    
    s = 2 / (torch.ones(ncat - 1, device=device) + torch.arange(ncat - 1, 0, -1, device=device).float())
    time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
            gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2).view(x.shape[0], -1).mean(dim=1).detach()

time_dependent_weights = time_dependent_cums / time_dependent_counts
time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()

plt.plot(torch.sqrt(time_dependent_weights.cpu()))
plt.savefig("timedependent_weight.png")

score_model = nn.DataParallel(ScoreNet(time_dependent_weights=torch.sqrt(time_dependent_weights)))
score_model = score_model.to(device)
score_model.train()

sampler = Euler_Maruyama_sampler
optimizer = Adam(score_model.parameters(), lr=lr)
torch.set_default_dtype(torch.float32)

## training steps

tqdm_epoch = range(num_epochs)
save_steps = 1
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    stime = time.time()
    for x in train_dataloader:
        random_t = torch.LongTensor(np.random.choice(np.arange(n_time_steps), size=x.shape[0],
                                                     p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                     time_dependent_weights).sum()).cpu().detach().numpy()))
        
        perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x.cpu(), random_t, v_one, v_one_loggrad)
        perturbed_x = perturbed_x.to(device) # [128,50,4]
        perturbed_x_grad = perturbed_x_grad.to(device)
        random_timepoints = timepoints[random_t].to(device)
        score = score_model(perturbed_x, random_timepoints) # torch.Size([128, 50, 4])

        s = 2 / (torch.ones(ncat - 1, device=device) + torch.arange(ncat - 1, 0, -1, device=device).float())
        
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        loss = torch.mean(torch.mean( 1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * 
                    s[(None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * 
                    (gx_to_gv(score, perturbed_x, create_graph=True)- gx_to_gv(perturbed_x_grad,perturbed_x)) ** 2, dim=(1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        
    print('Epoch {}: Average Loss: {:5f}'.format(epoch, avg_loss / num_items))
    # tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    
    if ( (epoch + 1) % save_steps == 0):
        torch.save(score_model.state_dict(), '../checkpoints/ddsm.pth')
        
        score_model.eval()
        torch.set_default_dtype(torch.float32)
        allsamples = []
        
        sep = 10
        sample_num = 10000
        for k in range(sep):
            allsamples.append(sampler(score_model,
                                      (50, 4),
                                      batch_size=int(sample_num/sep),
                                      max_time=4,
                                      min_time=4 / 400,
                                      time_dilation=1,
                                      num_steps=100,
                                      eps=1e-5,
                                      speed_balanced=speed_balanced,
                                      device=device,
                                      ).detach().cpu().numpy()
                              )
        allsamples = np.concatenate(allsamples, axis=0) # 10000, 50, 4
        allsamples = torch.tensor(allsamples, dtype=float)
        allsamples = onehot2seq(allsamples, seq_len=50)
        write_fa("../checkpoints/samples/epoch_{}.txt".format(epoch + 1), allsamples)
        score_model.train()



## sampling


score_model.load_state_dict(torch.load('../checkpoints/ddsm.pth'))
score_model.eval()
torch.set_default_dtype(torch.float32)

sep = 10
sample_num = 10000

for i in range(100):
    allsamples = []
    torch.manual_seed(i)
    for k in range(sep):
        allsamples.append(sampler(score_model,
                                  (50, 4),
                                  batch_size=int(sample_num/sep),
                                  max_time=4,
                                  min_time=4 / 400,
                                  time_dilation=1,
                                  num_steps=100,
                                  eps=1e-5,
                                  speed_balanced=speed_balanced,
                                  device=device,
                                  ).detach().cpu().numpy()
                          )
    allsamples = np.concatenate(allsamples, axis=0) # 10000, 50, 4
    allsamples = torch.tensor(allsamples, dtype=float)
    allsamples = onehot2seq(allsamples, seq_len=50)
    write_fa("../checkpoints/samples_200/seed_{}.txt".format(i), allsamples)
    print("samples for seed {} have been finished!".format(i))
