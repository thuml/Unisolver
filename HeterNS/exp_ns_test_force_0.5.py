import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params_icml2024 import get_args
from model_dict import get_model
from utils.adam import Adam
import math
import os

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

data_path = args.data_path

################################################################
# models
################################################################
model = get_model(args)
model.load_state_dict(torch.load(args.model_pretrain_path))
print(count_params(model))

################################################################
# load data and data normalization
################################################################

test_data = np.load(os.path.join(args.data_path, 'ns_vis_nu1e-05_m_0.5_u_200.npy'))
test_data = torch.from_numpy(np.concatenate([test_data], axis=0))

print(test_data.shape)

test_a = test_data[:, ::r1, ::r2, 10-T_in:10]
test_u = test_data[:, ::r1, ::r2, 10:10 + T_out]
print(test_u.shape)

test_a = test_a.reshape(ntest, s1, s2, T_in)

################################################################
# evaluation
################################################################

myloss = LpLoss(size_average=False)

step = 1
model.train()

test_l2_step = 0
test_l2_full = 0
index = 0


f = np.load(os.path.join(args.data_path, 'results_save_200/ns_vis_nu1e-05_m_0.5_f_200.npy'))
f = torch.from_numpy(np.concatenate([f], axis=0))[:, ::r1, ::r2]

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, f, test_u), batch_size=batch_size,
                                          shuffle=False)

path = 'vis_ns_other_force_0d5/'
if not os.path.exists(path):
    os.mkdir(path)
with torch.no_grad():
    for xx, f, yy in test_loader:
        index += 1
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx, torch.tensor([1e-5]).repeat(xx.shape[0]).cuda(), f.cuda())
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        
print(test_l2_step / ntest / (T_out / step),
        test_l2_full / ntest)
