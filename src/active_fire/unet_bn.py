import sys
import os 
import pandas as pd
from tqdm.auto import tqdm
import argparse
from datetime import datetime

import torch




from torchgeo.models import ResNet18_Weights, ResNet50_Weights

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lightning.pytorch import Trainer, seed_everything

from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torchmetrics.classification import BinaryConfusionMatrix


import torch.nn.functional as F

from my_dataset import MultiSatelliteDataset, calculate_mean_std

from torchmetrics import Precision, Recall, F1Score

import shutil 


seed_everything(42, workers=True)

GDAL_CROPED = True


LANDSAT_DATASET_PATH = '/dataset/Landsat/'
LANDSAT_ANNOTATION_FOLDER = 'GROUNDTRUTH_GABRIEL_patches_cp'
LANDSAT_IMG_FOLDER = 'IMG_all_patches'
LANDSAT_BANDS = (7,6,5)
# LANDSAT_BANDS = (1,2,3,4,5,6,7)

SENTINEL_DATASET_PATH = '/dataset/Sentinel/'
SENTINEL_ANNOTATION_FOLDER = 'mask1'
SENTINEL_IMG_FOLDER = 'imgs'
SENTINEL_BANDS = (6,5,4)

LANDSAT_DATAFRAME_PATH = f'/spectralgpt/active_fire/dataframes/folds-fogo/landsat-sentinel/landsat_extracted_folds.csv'
SENTINEL_DATAFRAME_PATH =  '/spectralgpt/active_fire/dataframes/folds-fogo/landsat-sentinel/sentinel_extracted_folds.csv'
LANDSAT_SENTINEL_DATAFRAME_PATH = '/spectralgpt/active_fire/dataframes/folds-fogo/landsat-sentinel/landsat-sentinel_folds.csv'



LOG_DIR = '/spectralgpt/downstream_tasks/unet/logs-unet-bn-gdal-croped'


if GDAL_CROPED:
    print(f'[INFO] Usando dados GDAL CROPED')
    SENTINEL_DATASET_PATH = '/dataset/Sentinel/gdal_croped/'
    SENTINEL_IMG_FOLDER = 'imgs_256'
    SENTINEL_ANNOTATION_FOLDER = 'annotations'
    SENTINEL_METHODS_FOLDER = 'methods'
    LOG_DIR += '-gdal-croped'

    SENTINEL_DATAFRAME_PATH =  '/spectralgpt/active_fire/dataframes/folds-fogo/landsat-sentinel/sentinel_extracted_folds_gdal_croped.csv'
    LANDSAT_SENTINEL_DATAFRAME_PATH = '/spectralgpt/active_fire/dataframes/folds-fogo/landsat-sentinel/landsat-sentinel_folds_gdal_croped.csv'


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Encoder, self).__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(base_channels * 8, base_channels * 16)
        
    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        c5 = self.bottom(p4)
        return c1, c2, c3, c4, c5

class Decoder(nn.Module):
    def __init__(self, base_channels=64):
        super(Decoder, self).__init__()
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        
    def forward(self, enc_feats, bottom):
        c1, c2, c3, c4 = enc_feats
        u4 = self.up4(bottom)
        merge4 = torch.cat([u4, c4], dim=1)
        d4 = self.conv4(merge4)
        u3 = self.up3(d4)
        merge3 = torch.cat([u3, c3], dim=1)
        d3 = self.conv3(merge3)
        u2 = self.up2(d3)
        merge2 = torch.cat([u2, c2], dim=1)
        d2 = self.conv2(merge2)
        u1 = self.up1(d2)
        merge1 = torch.cat([u1, c1], dim=1)
        d1 = self.conv1(merge1)
        out = self.out_conv(d1)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels)
        
    def forward(self, x):
        enc_feats = self.encoder(x)
        c1, c2, c3, c4, c5 = enc_feats
        out = self.decoder((c1, c2, c3, c4), c5)
        return out



    
class ActiveFireUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super(ActiveFireUNet, self).__init__()
        self.unet = UNet(in_channels, base_channels)
        
    def forward(self, x):
        seg_main = self.unet(x)  

        return seg_main
    

class SegNetModule(L.LightningModule):
    def __init__(self, model='unet-bn', encoder_name='default', decoder_name='default', num_classes=1, in_channels=3, weights=None, lr=1e-3, freeze_encoder=False):
        super().__init__()

        self.model_name = model
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.weights = weights
        self.lr = lr
        self.save_hyperparameters()

        self.model = ActiveFireUNet(in_channels=in_channels, num_classes=num_classes)


    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        
        # print(x)
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss' : loss}


    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)

        self.log('test_loss', loss, sync_dist=True)
        
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    

    def forward(self, x, domain_id=None):
        return {'backbone_prob' : self.model(x)} 

def build_datasets(args):
    print('[INFO] Preprando dataset landsat-sentinel')
    transform = None

    root_dir_map = {'landsat': LANDSAT_DATASET_PATH, 'sentinel': SENTINEL_DATASET_PATH}
    mask_folder_map = {'landsat': LANDSAT_ANNOTATION_FOLDER, 'sentinel': SENTINEL_ANNOTATION_FOLDER}
    img_folder_map = {'landsat': LANDSAT_IMG_FOLDER, 'sentinel': SENTINEL_IMG_FOLDER}
    bands = {'landsat': LANDSAT_BANDS, 'sentinel': SENTINEL_BANDS}
    
    quantification = {'landsat': args.quantification, 'sentinel': args.quantification}
    if isinstance(args.quantification, list):
        if len(args.quantification) == 1:
            quantification = {args.satellite: args.quantification[0]}
        elif len(args.quantification) == 2:
            quantification = {'landsat': args.quantification[0], 'sentinel': args.quantification[1]}

    mean_std = None
    if 'mean-std' in args.quantification:
        mean_std = define_means_stds(args)

    train_dataset = MultiSatelliteDataset(args.dataframe_path, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='train', transform=transform, means_stds=mean_std)
    val_dataset = MultiSatelliteDataset(args.dataframe_path, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='validation', transform=transform, means_stds=mean_std)


    test_dataset_full = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='test', transform=transform, means_stds=mean_std)
    test_dataset_landsat = MultiSatelliteDataset(LANDSAT_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='test', transform=transform, means_stds=mean_std)
    test_dataset_sentinel = MultiSatelliteDataset(SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='test', transform=transform, means_stds=mean_std)


    return train_dataset, val_dataset, test_dataset_full, test_dataset_landsat, test_dataset_sentinel



def define_means_stds(args):
    if type(args.quantification) != list:
        return None
    
    df_folds = pd.read_csv(args.dataframe_path)
    df_folds = df_folds[ (df_folds['fold'] == args.fold) & (df_folds['set'] == 'train') ]

    root_dir_map = {'landsat': LANDSAT_DATASET_PATH, 'sentinel': SENTINEL_DATASET_PATH}
    mask_folder_map = {'landsat': LANDSAT_ANNOTATION_FOLDER, 'sentinel': SENTINEL_ANNOTATION_FOLDER}
    img_folder_map = {'landsat': LANDSAT_IMG_FOLDER, 'sentinel': SENTINEL_IMG_FOLDER}
    

    df_folds['root_dir'] = df_folds['satellite'].apply(lambda x : root_dir_map[x])
    df_folds['img_folder'] = df_folds['satellite'].apply(lambda x : img_folder_map[x])
    df_folds['mask_folder'] = df_folds['satellite'].apply(lambda x : mask_folder_map[x])

    df_folds['image_path'] = df_folds.apply(lambda x : os.path.join(x['root_dir'], x['img_folder'], x['image']), axis=1)
    df_folds['mask_path'] = df_folds.apply(lambda x : os.path.join(x['root_dir'], x['mask_folder'], x['mask1']), axis=1)

    landsat_mean_std = None
    sentinel_mean_std = None
    if len(args.quantification) >= 1 and args.quantification[0] == 'mean-std':
        landsat_mean_std = calculate_mean_std(df_folds[ df_folds['satellite'] == 'landsat' ], 'image_path', LANDSAT_BANDS)
    
    if len(args.quantification) >= 2 and args.quantification[1] == 'mean-std':
        sentinel_mean_std = calculate_mean_std(df_folds[ df_folds['satellite'] == 'sentinel' ], 'image_path', SENTINEL_BANDS)
    

    

    if args.satellite == 'landsat':
        return {'landsat': landsat_mean_std}
    elif args.satellite == 'sentinel':
        return {'sentinel': sentinel_mean_std}
    elif args.satellite == 'landsat-sentinel':
        return {'landsat': landsat_mean_std, 'sentinel': sentinel_mean_std}



def evaluate_dataset(module, test_dataset, batch_size=1, workers=4):
    print('[INFO] Iniciando avaliação...')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)
    final_metrics = trainer.test(module, test_dataloader)
    print('[INFO] Avaliação finalizada')
    
    print(f'[INFO] Computando métricas...')
    cm_matrix = BinaryConfusionMatrix()    
    test_precision = Precision(task="binary")
    test_recall = Recall(task="binary")
    test_f1 = F1Score(task="binary")

    module.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images = batch['image'].to(module.device)
            masks_batch = batch['mask'].to(module.device)
            # masks_batch = masks_batch.unsqueeze(1)  # Adiciona a dimensão de canal

            y_hat = module.model(images)
            # y_hat = torch.sigmoid(y_hat).squeeze(1)  # Remove a dimensão de canal
            y_hat = torch.sigmoid(y_hat)
            y_hat = y_hat > 0.5

            test_precision.update(y_hat, masks_batch.int())
            test_recall.update(y_hat, masks_batch.int())
            test_f1.update(y_hat, masks_batch.int())
            cm_matrix.update(y_hat, masks_batch.int())

    precision = test_precision.compute()
    recall = test_recall.compute()
    f1 = test_f1.compute()
    cm = cm_matrix.compute()
    
    module.train()

    return precision, recall, f1, cm, final_metrics






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--satellite', choices=['sentinel', 'landsat', 'landsat-sentinel'], type=str, required=True)
    parser.add_argument('--quantification', default=1.0, nargs='+')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--dataframe-path', default=LANDSAT_SENTINEL_DATAFRAME_PATH, type=str)

    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='Máx number of total epochs to run')
    parser.add_argument('-p', '--early-stopping-patience', default=10, type=int)
    parser.add_argument('-j', '--workers', default=-2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-d', '--devices', default=None, type=int, help='Number of devices to use')
    parser.add_argument('--model', default='unet', type=str)
    parser.add_argument('--encoder', default='resnet50', type=str)
    parser.add_argument('--weights', default=None, type=str)    
    parser.add_argument('--num-classes', default=1, type=int)
    parser.add_argument('--freeze-encoder', action="store_true")
    parser.add_argument('--log-dir', default=LOG_DIR, type=str, help='log directory')


    args = parser.parse_args()

    weights_id = args.weights
    if weights_id is None:
        weights_id = 'scratch'
    elif os.path.exists(weights_id):
        weights_id = os.path.basename(weights_id).split('.')[0]
    
    bands_id = ''
    if args.satellite == 'landsat':
        bands_id = ''.join(list(map(str, LANDSAT_BANDS)))
    elif args.satellite == 'sentinel':
        bands_id = ''.join(list(map(str, SENTINEL_BANDS)))
    elif args.satellite == 'landsat-sentinel':
        bands_id = ''.join(list(map(str, LANDSAT_BANDS + SENTINEL_BANDS)))

    quantification_id = ''.join(list(map(str, list(args.quantification))))

    base_folder_name = f'{args.model}-{args.encoder}-{weights_id}-b{bands_id}-q{quantification_id}-e{args.epochs}-lr{args.lr}-bs{args.batch_size}'
    
    if args.freeze_encoder:
        log_dir = os.path.join(args.log_dir, args.satellite, f'{base_folder_name}-freeze-encoder', str(args.fold))
    else:
        log_dir = os.path.join(args.log_dir, args.satellite, base_folder_name, str(args.fold))
    
    # log_dir += f'loss-weights-b{args.w_loss_backbone}-dt{args.w_loss_dt}-fused{args.w_loss_fused}'

    os.makedirs(log_dir, exist_ok=True)

    start_at = datetime.now()

    with open(os.path.join(log_dir, 'args.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write(f'Started at: {start_at}\n')
        f.write(str(args))
        f.write('\n')

    print(f'[INFO] Preprando dataset {args.satellite}...')
    transform = None


    train_dataset, val_dataset, test_dataset_full, test_dataset_landsat, test_dataset_sentinel = build_datasets(args)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.workers, 
        drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers)

    print('[INFO] Criando modelo...')
    print(f'Encoder: {args.encoder}')
    print(f'Carregando pesos prétreinados...')

    in_channels = max(len(LANDSAT_BANDS), len(SENTINEL_BANDS))
    weights = None
    if args.weights is not None:
        if type(args.weights) == str and os.path.exists(args.weights):
            weights = args.weights
        else:
            if args.encoder == 'resnet18':
                weights = ResNet18_Weights[args.weights]
                in_channels = weights.meta['in_chans']
            elif args.encoder == 'resnet50':
                weights = ResNet50_Weights[args.weights]
                in_channels = weights.meta['in_chans']
            else:
                weights = args.weights
    
    
    module = SegNetModule(model=args.model, encoder_name=args.encoder, num_classes=args.num_classes, in_channels=in_channels, weights=weights, lr=args.lr, freeze_encoder=args.freeze_encoder)

        
    logger = TensorBoardLogger(save_dir=os.path.join(log_dir, 'tensorboard'), name='lightning_logs')

    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=int(args.early_stopping_patience), verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(log_dir, 'checkpoints'), save_top_k=1, monitor="valid_loss", verbose=False, mode="min")


    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(default_root_dir=log_dir, check_val_every_n_epoch=1, log_every_n_steps=1, accelerator=accelerator, max_epochs=args.epochs, devices=[args.devices], logger=logger, callbacks=[early_stop_callback, checkpoint_callback])

    print('[INFO] Iniciando treinamento...')
    trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print('[INFO] Treinamento finalizado')

    # Carrega os melhores pesos do early stopping
    best_ckpt_path = checkpoint_callback.best_model_path

    if best_ckpt_path and os.path.exists(best_ckpt_path):
        print(f'[INFO] Melhor modelo encontrado em: {best_ckpt_path}')
        
        final_path = os.path.join(log_dir, 'checkpoints', 'final_model.ckpt')
        
        shutil.copy(best_ckpt_path, final_path)
        print(f'[INFO] Cópia do melhor modelo salva em: {final_path}')
        
        module = SegNetModule.load_from_checkpoint(best_ckpt_path)
        print('[INFO] Pesos carregados na memória para uso imediato.')

    else:
        print('[WARN] Nenhum checkpoint de melhor modelo encontrado. Salvando o estado atual.')
        trainer.save_checkpoint(os.path.join(log_dir, 'checkpoints', 'final_model.ckpt'))


    print(f'[INFO] Avaliando dataset Landsat-Sentinel - FOLD {args.fold}...')
    
    end_at = datetime.now()
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(module, test_dataset_full, batch_size=args.batch_size, workers=args.workers)

    print(f'[INFO] Salvando resultados...')
    with open(os.path.join(log_dir, 'final_results.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write('Args:\n')
        f.write(str(args))
        f.write('\n')
        f.write(f'Started at: {start_at}\n')
        f.write(f'Final results at: {end_at}\n')
        f.write('Dataset: Landsat-Sentinel\n')
        f.write(str(final_metrics))
        f.write(f'\nSklearn Metrics:\n')
        f.write(f'Precision: {str(precision)}\n')	
        f.write(f'Recall: {str(recall)}\n')
        f.write(f'F1: {str(f1)}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(str(cm_matrix))
        f.write('\n')

    
    print(f'[INFO] Avaliando dataset Landsat...')
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(module, test_dataset_landsat, batch_size=args.batch_size, workers=args.workers)

    print(f'[INFO] Salvando resultados...')
    with open(os.path.join(log_dir, 'final_results.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write('Dataset: Landsat\n')
        f.write(str(final_metrics))
        f.write(f'\nSklearn Metrics:\n')
        f.write(f'Precision: {str(precision)}\n')	
        f.write(f'Recall: {str(recall)}\n')
        f.write(f'F1: {str(f1)}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(str(cm_matrix))
        f.write('\n')

    print(f'[INFO] Avaliando dataset Sentinel...')
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(module, test_dataset_sentinel, batch_size=args.batch_size, workers=args.workers)

    print(f'[INFO] Salvando resultados...')
    with open(os.path.join(log_dir, 'final_results.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write('Dataset: Sentinel\n')
        f.write(str(final_metrics))
        f.write(f'\nSklearn Metrics:\n')
        f.write(f'Precision: {str(precision)}\n')	
        f.write(f'Recall: {str(recall)}\n')
        f.write(f'F1: {str(f1)}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(str(cm_matrix))
        f.write('\n')

        
    
    with open(os.path.join(log_dir, 'args.txt'), 'a+') as f:
        f.write(f'Ended at: {end_at}\n')
        f.write(f'Time elapsed: {end_at - start_at}\n')
        f.write('-'*80 + '\n')
        f.write('\n')

    




