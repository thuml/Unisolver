import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange


def get_bc_dataset(args, dataset_file):
    processed_data_path = os.path.splitext(dataset_file)[0] + '_processed.npy'
    if os.path.exists(processed_data_path):
        dataset = np.load(processed_data_path)
    else:
        data = np.load(dataset_file)[:, ::args.h_down, ::args.w_down] # 10000, args.h/args.h_down, args.w/args.w_down
        dataset = []
        for i in range(args.ntotal): 
            dataset.append(data[i:i + args.T_in + args.T_out, :, :].transpose(1, 2, 0))
        dataset = np.stack(dataset, axis=0) # N, 512/4, 512/4, 10+10
        np.save(processed_data_path, dataset)
    dataset = torch.from_numpy(dataset.astype(np.float32))
    return dataset


class SmokeDataset(Dataset):
    def __init__(self, args, data_path, split='train'):
        self.data_path = data_path
        self.h = int(((args.h - 1) / args.h_down) + 1)
        self.w = int(((args.w - 1) / args.w_down) + 1)
        self.z = int(((args.z - 1) / args.z_down) + 1)
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain

        self.r1=args.h_down
        self.r2 = args.w_down
        self.r3=args.z_down

        self.split = split
        self.length = args.ntrain if split=='train' else args.ntest

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # index_tmp = index if self.split=='train' else self.ntrain + index
        index_tmp = index 
        data_file = Path(self.data_path) / f'smoke_{index_tmp}.npz'
        data = np.load(data_file)
        data = torch.cat([torch.from_numpy(data['fluid_field']), torch.from_numpy(data['velocity'])], dim=-1)  # 20 32 32 32 1+3=4
        input_x = rearrange(data[:self.T_in], 't h w z c -> h w z c t')
        input_x = input_x[::self.r1,::self.r2,::self.r3,:,:]
        input_y = rearrange(data[self.T_in:self.T_in+self.T_out], 't h w z c -> h w z c t')
        input_y = input_y[::self.r1,::self.r2,::self.r3,:,:]
        return input_x, input_y # B Z H W C T
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False, num_workers=10)



class SeaDataset(Dataset):
    '''
    ['thetao', 'so', 'uo', 'vo', 'zos'] (b, 5, 180, 300)
    thetao: sea_water_potential_temperature, range [4.537, 26.204]
    so: sea_water_salinity, range [32.241, 35.263]
    uo: eastward_sea_water_velocity, range [-0.893, 1.546]
    vo: northward_sea_water_velocity, range [-1.646, 1.088]
    zos: sea_surface_height_above_geoid, range [-0.342, 1.511]
    '''
    def __init__(self, args, region='kuroshio', split='train', var=None):
        self.data_path = args.data_path # /data/sea_data_small/data_sea
        self.data_files = sorted([data_file for data_file in os.listdir(self.data_path) if region in data_file])
        self.fill_value = args.fill_value
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        var_dict = {'thetao':0, 'so':1, 'uo':2, 'vo':3, 'zos':4}
        if var is not None and var not in var_dict.keys():
            raise Exception('var must be None or one of [\'thetao\', \'so\', \'uo\', \'vo\', \'zos\']')
        elif var is not None:
            self.var_idx = var_dict[var]
        else:
            self.var_idx = None

        self.split = split
        self.length = args.ntrain if split=='train' else args.ntest
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_tmp = index if self.split=='train' else self.ntrain + index
        input_x, input_y = [], []
        for i in range(self.T_in + self.T_out):
            data_file = Path(self.data_path) / self.data_files[index_tmp + i]
            data = np.load(data_file).astype(np.float32) # c=5 h=180 w=300 
            data[data <= self.fill_value] = 0.
            data[1][data[1] == 0] = 30. # so \in (32.241, 35.263)
            if i < self.T_in:
                input_x.append(torch.from_numpy(data))
            else:
                input_y.append(torch.from_numpy(data))
        if self.var_idx is None:
            input_x = torch.stack(input_x, dim=-1) # c h w t=10
            input_y = torch.stack(input_y, dim=-1)
            input_x = rearrange(input_x, 'c h w t -> h w (t c)')
            input_y = rearrange(input_y, 'c h w t -> h w (t c)')
        else:
            input_x = torch.stack(input_x, dim=-1)[self.var_idx] # h w t=10
            input_y = torch.stack(input_y, dim=-1)[self.var_idx]
        return input_x, input_y # B H W T*C
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)