import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
import math
import os
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
gpu_num = 3
    
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

ntrain = 15000 // gpu_num
ntest = 200 // gpu_num
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = 10
T_out = 10

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

train_data = torch.from_numpy(np.load(os.path.join(args.data_path, 'u_5_nu_3_force_15000_64.npy')))[:, ::r1, ::r2, :]

def get_force():
    f = torch.zeros([1,s1,s2])
    t = torch.linspace(0, 1, s1+1, device=device)
    t = t[0:-1]
    X,Y = torch.meshgrid(t,t)
    f[0] = 0.1*(torch.sin(2*np.pi*(X + Y)) + torch.cos(2*np.pi*(X + Y)))
                
    return f

vis_train = torch.from_numpy(np.load(os.path.join(args.data_path, 'vis_5_nu_3_force_15000_64.npy')))
f = torch.from_numpy(np.load(os.path.join(args.data_path, 'f_5_nu_3_force_15000_64.npy')))[:, ::r1, ::r2]
print(f.shape)

vis_test = torch.tensor([[0.00001]]).repeat(1, 200).flatten()
f_test = get_force()[0:1].repeat(1, 200, 1, 1).reshape(200, 64, 64)

################################################################
# load data and data normalization
################################################################


train_dataset = torch.utils.data.TensorDataset(train_data, vis_train, f)
train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True) # B H W 20

TEST_PATH = os.path.join(args.data_path, 'ns_5forces_3viscosity_test.npy')
test_data = np.load(TEST_PATH)
test_data = torch.from_numpy(test_data)

test_a = test_data[2000:2200, :, :, :T_in]
test_u = test_data[2000:2200, :, :, T_in:]
print(test_u.shape)

test_batch_size = 200//gpu_num
test_dataset = torch.utils.data.TensorDataset(test_a,vis_test, f_test, test_u)
test_sampler = DistributedSampler(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, sampler=test_sampler, drop_last=True)
################################################################
# models
################################################################
model = get_model(args)
model=torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

myloss = LpLoss(size_average=False)
myloss_wo_reduce = LpLoss(size_average=False, reduction=False)

step = 1
M = 5
lambda_reg = 0.01
from tqdm import tqdm
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    train_loader.sampler.set_epoch(ep)
    sample = 0
    for xx, mu, f in tqdm(train_loader): # B H W 20
        sample += 1
        loss = 0
        xx = xx.to(device)
        mu = mu.to(device)
        f = f.to(device)

        for t in range(0, xx.shape[-1]-T_in , 1):
            y = xx[..., t+T_in:t+T_in+1]
            im = model(xx[..., t:t+T_in], mu, f)
            loss = myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            pred = im
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l2_step += loss.item()


    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx,mu,f, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            mu = mu.to(device)
            f = f.to(device)
            for t in range(0, yy.shape[-1], step):
                y = yy[..., t:t + step]
                im = model(xx, mu, f)
                loss += myloss(im.reshape(test_batch_size, -1), y.reshape(test_batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(test_batch_size, -1), yy.reshape(test_batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
          test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest)
    writer.add_scalar('train_l2_step', train_l2_step / (ntrain * (20 - T_in)), ep)
    writer.add_scalar('test_l2_step', test_l2_step / ntest / (T_out / step), ep)
    writer.add_scalar("test_l2_full", test_l2_full / ntest, ep)
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        if local_rank == 0:
            torch.save(model.module.state_dict(), os.path.join(model_save_path, model_save_name+str(ep)+'.pt'))
