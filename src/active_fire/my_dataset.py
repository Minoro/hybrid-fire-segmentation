import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
# from osgeo import gdal
import rasterio
import skimage.io as io
from imgaug import augmenters as iaa
import torchvision.transforms.functional as transF

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from torchvision.transforms import v2


import pandas as pd
from tqdm.auto import tqdm


class MultiSatelliteDataset(Dataset):
    
    def __init__(self, csv_file, root_dir_map, mask_folder_map, img_folder_map, bands_map, quantification_map, fold=1, set='train', transform=None, means_stds=None):
        self.df_folds = pd.read_csv(csv_file)
        self.df_folds = self.df_folds[ (self.df_folds['fold'] == fold) & (self.df_folds['set'] == set) ]

        self.df_folds['root_dir'] = self.df_folds['satellite'].apply(lambda x : root_dir_map[x])
        self.df_folds['img_folder'] = self.df_folds['satellite'].apply(lambda x : img_folder_map[x])
        self.df_folds['mask_folder'] = self.df_folds['satellite'].apply(lambda x : mask_folder_map[x])
        
        self.df_folds['image_path'] = self.df_folds.apply(lambda x : os.path.join(x['root_dir'], x['img_folder'], x['image']) if x['annotation'] != 'seamline' else x['image'], axis=1)
        self.df_folds['mask_path'] = self.df_folds.apply(lambda x : os.path.join(x['root_dir'], x['mask_folder'], x['mask1']) if x['annotation'] != 'seamline' else x['mask1'], axis=1)

        # self.df_folds = self.df_folds[ self.df_folds['num_fire_pixels'] > 0 ]

        self.mask_folder_map = mask_folder_map
        self.bands_map = bands_map
        self.transform = transform
        self.quantification_map = quantification_map
        self.means_stds = means_stds
        
    def __len__(self):
        return len(self.df_folds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        satellite = self.df_folds.iloc[idx]['satellite']
        quantification = self.quantification_map[satellite]        
        img_path = self.df_folds.iloc[idx]['image_path']
        annotation = self.df_folds.iloc[idx]['annotation']
        with rasterio.open(img_path) as src:
            # img = src.read((6,5,4))
            if annotation == 'seamline':
                img = src.read()
            else:
                img = src.read(self.bands_map[satellite])
        
        # img = np.clip(img/8160.0, 0, 1) # Normalização para o Satlass
        # img = img/self.quantification

        # Trata os pixels saturados
        if satellite == 'modis':
            img[(img >= 65500) & (img <= 65535) ] = 0

        img =  img.astype(np.float32)
        if isinstance(quantification, str) and not quantification.replace('.','',1).isnumeric():
            quantification = quantification.strip()
            if quantification == 'auto' or quantification == 'min-max':

                img = img.transpose(1,2,0)
                kid = (img - img.min(axis=(0, 1), keepdims=True))
                mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
                img = kid / (mom + 1e-10)
                img = img.transpose(2,0,1)

            elif quantification == 'img-mean-std':
                img = (img - img.mean()) / (img.std() + 1e-10)

            elif quantification == 'mean-std':
                img = img.transpose(1,2,0)
                img = (img - self.means_stds[satellite]['mean']) / self.means_stds[satellite]['std']
                img = img.transpose(2,0,1)

            elif quantification == 'satlas':
                img = np.clip(img/8160.0, 0, 1)
            elif quantification == 'raw':
                pass
            else:
                raise ValueError(f'Invalid quantification value, it must be a number or "auto" or "raw". Informed value: {quantification}')
        else:   
            img = img/float(quantification)
            

        mask_path = self.df_folds.iloc[idx]['mask_path']
        if satellite == 'modis':
            # Considera o nível mímino de fogo
            with rasterio.open(mask_path) as src:
                mask = (src.read() >= 7)
        else:
            with rasterio.open(mask_path) as src:
                mask = (src.read() != 0)
        
        # mask =  mask.astype(np.float32)

        if self.transform:
            img = torch.from_numpy(img.astype(np.float32).copy())
            mask = torch.from_numpy(mask.astype(np.float32))
        
            img, mask = self.transform(img, mask)
        else:
            img = torch.from_numpy(img.astype(np.float32).copy())
            # mask = torch.from_numpy(mask.astype(np.int32)).long()
            mask = torch.from_numpy(mask.astype(np.float32))
        
        # mask = mask.squeeze(0)
        if satellite == 'landsat':
            sat_encoding = torch.tensor([1.0, 0.0])
            domain_id = 0
        elif satellite == 'sentinel':
            sat_encoding = torch.tensor([0.0, 1.0])
            domain_id = 1
            
        return {'image': img, 'mask': mask, 'satellite': sat_encoding, 'domain_id': domain_id}





def calculate_mean_std(df, image_column, bands=None):
    means = []
    stds = []
    
    for image_path in tqdm(df[image_column]):
        with rasterio.open(image_path) as src:
            image = src.read(bands)
            means.append(np.mean(image, axis=(1, 2)))
            stds.append(np.std(image, axis=(1, 2)))
    
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    return mean, std
