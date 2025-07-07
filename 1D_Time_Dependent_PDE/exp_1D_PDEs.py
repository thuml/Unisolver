import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
import numpy as np
import math
import os
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import h5py
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
    
################################################################
# configs
################################################################
args = get_args()

# DDP init
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

from torch.utils.tensorboard import SummaryWriter
path = args.log_path
writer = SummaryWriter(path)
print('logging training state to ' + path)
if not os.path.exists(path):
    os.makedirs(path)

gpu_num = 1
real_ntrain = 3000000
ntrain = real_ntrain // gpu_num
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
s1 = int(((args.h - 1) / r1) + 1)
T_in = 1
T_out = 100


filename_list = [
"/data/pdeformer_data/custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed100.hdf5"
]

import pickle
with open('embeddings_dict.pkl', 'rb') as file:
    loaded_embeddings_dict = pickle.load(file)

def get_llm_embeddings(file):
    embeddings = []
    kappa_type = np.array(file['coef']['kappa']['coef_type_all']).squeeze()
    s_type = np.array(file['coef']['s']['coef_type_all']).squeeze()
    f0 = np.array(file['coef']['f0_all']).squeeze()
    f1 = np.array(file['coef']['f1_all']).squeeze()
    f0[f0!=0]=1
    f1[f1!=0]=1

    f0 = f0.astype(np.int32)
    f1 = f1.astype(np.int32)
    embeddings = []
    from tqdm import tqdm
    print(f0.shape)
    for i in tqdm(range(10000)):
        key = (f0[i, 1],f0[i, 2],f0[i, 3], f1[i, 1],f1[i, 2],f1[i, 3], s_type[i], kappa_type[i])
        embeddings.append(loaded_embeddings_dict[key])
        
    return np.stack(embeddings, axis=0)
class INR_Dataset(Dataset):
    def __init__(self, filename_list):
        super().__init__()
        solution_fields = []
        f0s = []
        f1s = []
        s_fields = []
        kappa_fields = []
        boundary_values = []
        llm_embeddings = []
        self.x_coord = None
        self.t_coord = None
        self.filename_list = filename_list
        for filename in filename_list:
            with h5py.File(filename, 'r') as file:
                llm_embeddings.append(get_llm_embeddings(file))
                if self.x_coord is None:
                    self.x_coord = np.array(file['x_coord'])
                if self.t_coord is None:
                    self.t_coord = np.array(file['t_coord'])
                solution_field = np.array(file['u_sol_all'])
                kappa_field = np.array(file['coef']['kappa']['field_all'])
                s_field = np.array(file['coef']['s']['field_all'])
                f0 = np.array(file['coef']['f0_all'])
                f1 = np.array(file['coef']['f1_all'])
                
                solution_fields.append(solution_field)
                kappa_fields.append(kappa_field)
                s_fields.append(s_field)
                f0s.append(f0)
                f1s.append(f1)
                
                if 'circ' in filename:
                    boundary_values.append(np.array([[-1, 0, -1, 0, 0, -1, 0, -1, 0, 0]]).repeat(10000, axis=0))
                else:
                    l_type = np.array(file['coef/bc_l/bc_type_all'])
                    l_value = np.array(file['coef/bc_l/bc_val_all'])
                    l_value_type = np.array(file['coef/bc_l/bc_val_type_all'])
                    l_dx_coef = np.array(file['coef/bc_l/dx_u_coef_all'])
                    l_u_coef = np.array(file['coef/bc_l/u_coef_all'])
                    r_type = np.array(file['coef/bc_r/bc_type_all'])
                    r_value = np.array(file['coef/bc_r/bc_val_all'])
                    r_value_type = np.array(file['coef/bc_r/bc_val_type_all'])
                    r_dx_coef = np.array(file['coef/bc_r/dx_u_coef_all'])
                    r_u_coef = np.array(file['coef/bc_r/u_coef_all'])
                    boundary_values.append(
                        np.stack([
                            l_type,
                            l_value,
                            l_value_type,
                            l_dx_coef,
                            l_u_coef,
                            r_type,
                            r_value,
                            r_value_type,
                            r_dx_coef,
                            r_u_coef
                        ], axis=-1)
                    )
        
        self.llm_embeddings = np.concatenate(llm_embeddings, axis=0).astype(np.float32)
        self.solution_fields = np.concatenate(solution_fields, axis=0).astype(np.float32)
        self.kappa_fields = np.concatenate(kappa_fields, axis=0).astype(np.float32)
        self.s_fields = np.concatenate(s_fields, axis=0).astype(np.float32)
        self.f0s = np.concatenate(f0s, axis=0).squeeze().astype(np.float32)
        self.f1s = np.concatenate(f1s, axis=0).squeeze().astype(np.float32)
        self.boundary_values = np.concatenate(boundary_values).squeeze().astype(np.float32)
        
        X, T = np.meshgrid(self.x_coord, self.t_coord[1:])
        self.t_x_coord = np.stack([X, T], axis=-1).reshape(-1, 2).astype(np.float32) # Time, Resolution
        self.num_tx_samp_pts = 8192
        
    def __getitem__(self, index):
        sol = self.solution_fields[index]
        kappa = self.kappa_fields[index]
        s = self.s_fields[index]
        f0 = self.f0s[index]
        f1 = self.f1s[index]
        bc = self.boundary_values[index]
        llm = self.llm_embeddings[index]
        
        ic = sol[0]
        result = sol[1:].reshape(-1, 1)
        tx_sample_idx = np.random.randint(0, self.t_x_coord.shape[0], self.num_tx_samp_pts)
        return ic, kappa, s, f0, f1, bc, self.t_x_coord[tx_sample_idx], result[tx_sample_idx], llm
        
    
    def __len__(self):
        return 10000 * len(self.filename_list)
        

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name


################################################################
# load data and data normalization
################################################################

train_dataset = INR_Dataset(filename_list)
train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True) 

################################################################
# models
################################################################
model = get_model(args)
model=torch.nn.parallel.DistributedDataParallel(model)
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

myloss = LpLoss(size_average=False)
myloss_wo_reduce = LpLoss(size_average=False, reduction=False)

infer_step = 100
from tqdm import tqdm
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    train_loader.sampler.set_epoch(ep)
    sample = 0
    for ic, kappa, s, f0, f1, bc, grid, sol, llm in tqdm(train_loader): # B H W 20
        sample += 1
        loss = 0
        ic = ic.to(device)
        kappa = kappa.to(device)
        s = s.to(device)
        f0 = f0.to(device)
        f1 = f1.to(device)
        bc = bc.to(device)
        grid = grid.to(device)
        sol = sol.to(device)
        llm = llm.to(device)

        im = model(ic, kappa, s, f0, f1, bc, grid, llm)
        loss = myloss(im.reshape(batch_size, -1), sol.reshape(batch_size, -1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_l2_step += loss.item()


    test_l2_step = 0
    test_l2_full = 0
    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / infer_step), train_l2_full / ntrain)
    writer.add_scalar('train_l2_step', train_l2_step / ntrain / (T_out / infer_step), ep)
    if ep % step_size == 0:
        if local_rank == 0:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.module.state_dict(), os.path.join(model_save_path, model_save_name+str(ep)+'.pt'))
