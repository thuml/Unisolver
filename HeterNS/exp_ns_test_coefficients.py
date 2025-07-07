import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
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


TEST_PATH = os.path.join(args.data_path, 'u_64.npy')
TEST_PATH_NU = os.path.join(args.data_path, 'vis_64.npy')
TEST_PATH_F = os.path.join(args.data_path, 'f_64.npy')

ntest = 200
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = 200
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

################################################################
# models
################################################################
model = get_model(args)
state_dict = torch.load(args.model_pretrain_path)
model.load_state_dict(state_dict)
print(count_params(model))
################################################################
# load data and data normalization
################################################################
raw_data = torch.from_numpy(np.load(TEST_PATH)).cuda()
raw_data_nu = torch.from_numpy(np.load(TEST_PATH_NU)).cuda()
raw_data_f = torch.from_numpy(np.load(TEST_PATH_F)).cuda()

test_a = raw_data[:, ::r1, ::r2, :T_in]
test_u = raw_data[:, ::r1, ::r2, T_in:T_in + T_out]
test_nu = raw_data_nu
test_f = raw_data_f[:, ::r1, ::r2]
myloss = LpLoss(size_average=False)
step = 1
visual_path = os.path.join('vis_ns', args.model + '_16200')
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
    
with torch.no_grad():
    index_start = 0
    for viscosity in range(27):
        for force in range(3):
            viscosity = str(viscosity)
            force = str(force)
            test_l2_step = 0
            test_l2_full = 0
            index = 0
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                test_a[index_start:index_start+200, :, :, :], test_nu[index_start:index_start+200],
                test_f[index_start:index_start+200, :, :], test_u[index_start:index_start+200, :, :, :]
                ), batch_size=batch_size, shuffle=False)
            path = os.path.join(visual_path, viscosity + '_' + force)
            if not os.path.exists(path):
                os.makedirs(path)
            for xx, mu, f, yy in test_loader:
                index += 1
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                mu = mu.to(device)
                f = f.to(device)

                for t in range(0, T_out, step):
                    y = yy[..., t:t + step]
                    im = model(xx, mu, f)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)
                    plt.figure(figsize=(20, 8), dpi=200) 
                    plt.subplot(1, 3, 1)
                    plt.imshow(im[1,:, :, 0].detach().cpu().numpy())
                    plt.colorbar(shrink=0.75)
                    plt.subplot(1, 3, 2)
                    plt.imshow(y[1, :, :, 0].detach().cpu().numpy())
                    plt.colorbar(shrink=0.75)
                    plt.subplot(1, 3, 3)
                    plt.imshow(abs(im - y)[1, :, :, 0].detach().cpu().numpy())
                    plt.colorbar(shrink=0.75)           
                    plt.tight_layout(pad=0.5)         
                    plt.savefig(os.path.join(path, str(index) + '_' + str(t)))
                    plt.close()

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
            print(viscosity, force, test_l2_step / ntest / (T_out / step), test_l2_full / ntest)
            index_start += 200
