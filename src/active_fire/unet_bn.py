import sys
import os
import torch
import torch.nn as nn

from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics import Precision, Recall, F1Score

from torch import nn
import lightning as L
import torchvision.transforms as T

from lightning.pytorch import Trainer, seed_everything


from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from tqdm.auto import tqdm

from my_dataset import MultiSatelliteDataset

from datetime import datetime

import json

import numpy as np
from typing import Dict, Any

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.ndimage import label

import shutil

from dafdm import DAFDM
from spectralgpt.src.models_vit_tensor_CD_2 import vit_base_patch8_256_channel3
from spectralgpt.util.pos_embed import interpolate_pos_embed

from argparse import ArgumentParser

seed_everything(42, workers=True)

# None or Unet
BACKBONE = 'Unet'

OVERRIDE_TRAINED = False

USE_FILM = False
NUM_SENSORS_THREHOLD_EQUATIONS = 2


FOLD = 1
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 50
ES_PATIENCE = 10
WORKERS = 12
DEVICES = [1]

SPECTRALGPT_WEIGHTS_PATH = None

LANDSAT_DATASET_PATH = '/dataset/Landsat/'
LANDSAT_ANNOTATION_FOLDER = 'manual_annotations_patches'
LANDSAT_IMG_FOLDER = 'landsat_patches'
LANDSAT_BANDS = (7,6,5)
LANDSAT_QUANTIFICATION = 65535.0

SENTINEL_DATASET_PATH = '/dataset/Sentinel/'
SENTINEL_IMG_FOLDER = 'imgs'
SENTINEL_ANNOTATION_FOLDER = 'mask1'
SENTINEL_BANDS = (6,5,4)
SENTINEL_QUANTIFICATION = 10000.0

LANDSAT_DATAFRAME_PATH = f'/hybrid-fire-segmentation/resources/dataframes/folds/landsat-sentinel/landsat_extracted_folds.csv'
SENTINEL_DATAFRAME_PATH =  '/hybrid-fire-segmentation/resources/dataframes/folds/landsat-sentinel/sentinel_extracted_folds.csv'
LANDSAT_SENTINEL_DATAFRAME_PATH = '/hybrid-fire-segmentation/resources/dataframes/folds/landsat-sentinel/landsat-sentinel_folds.csv'


LOG_DIR = '/hybrid-fire-segmentation/resources/unet/logs/'

class RandomFlipRotate:

    def __init__(self, degrees=(0, 360), p_flip=0.5):
        self.rotation = T.RandomRotation(degrees)
        self.h_flip = T.RandomHorizontalFlip(p=p_flip)
        self.v_flip = T.RandomVerticalFlip(p=p_flip)

    def __call__(self, image, mask):
        angle = self.rotation.get_params(self.rotation.degrees)
        image = T.functional.rotate(image, angle)
        mask = T.functional.rotate(mask, angle)

        if torch.rand(1).item() < self.h_flip.p:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)

        if torch.rand(1).item() < self.v_flip.p:
            image = T.functional.vflip(image)
            mask = T.functional.vflip(mask)

        return image, mask
    



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat((x, skip_features), dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        
        b = self.bottleneck(p4)
        
        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)
        
        # outputs = torch.sigmoid(self.final_conv(d4))
        outputs = self.final_conv(d4)
        return outputs



def sigmoid_stable_np(x):
                x = np.clip(x, -20, 20)
                return 1.0 / (1.0 + np.exp(-x))



class LitDTBranchAdvanced(L.LightningModule):
    def __init__(self, backbone='Unet', lr=1e-3, alpha=0.5, loss_b_weight=1.0, loss_dt_weight=1.0, loss_fused_weight=1.0):
        super().__init__()
        self.backbone_type = backbone
        self.lr = lr
        self.loss_b_weight = loss_b_weight
        self.loss_dt_weight = loss_dt_weight
        self.loss_fused_weight = loss_fused_weight
        
        self.save_hyperparameters()

        if self.backbone_type is None:
            self.backbone = None
        elif self.backbone_type.lower() == 'unet':
            self.backbone = UNet(in_channels=3, out_channels=1)
        elif self.backbone_type.lower() == 'dafdm':
            self.backbone = DAFDM(in_channels=3, out_channels=1) 
        elif self.backbone_type.lower() == 'spectralgpt':
            self.backbone = create_spectralgpt(SPECTRALGPT_WEIGHTS_PATH)
        else:
            raise ValueError(f'Backbone {self.backbone_type} não implementada.')

        self.loss_fn = nn.BCEWithLogitsLoss()
    
        self.alpha = nn.Parameter(torch.tensor(alpha)) 

    def forward(self, x, domain_id=None):
      
        
        logits_unet = self.backbone(x)
        probs_unet = torch.sigmoid(logits_unet)
        
        return {
            'backbone_logits': logits_unet,
            'backbone_prob': probs_unet,
        }

    def training_step(self, batch, batch_idx):
        x, y, domain_id = batch['image'], batch['mask'], batch['domain_id']
        out = self(x, domain_id)
        
        
        
        loss = self.loss_fn(out['backbone_logits'], y)
        
        
        f1 = self.f1_score(y, torch.sigmoid(out['backbone_prob']))
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, domain_id = batch['image'], batch['mask'], batch['domain_id']
        out = self(x, domain_id)

        loss = self.loss_fn(out['backbone_logits'], y)
        

        f1 = self.f1_score(y, torch.sigmoid(out['backbone_prob']))
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, domain_id = batch['image'], batch['mask'], batch['domain_id']
        out = self(x, domain_id)

       

        loss = self.loss_fn(out['backbone_logits'], y)



        f1 = self.f1_score(y, torch.sigmoid(out['backbone_prob']))
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        
    
    def __call__(self, x, domain_id=None):
        return self.forward(x, domain_id=domain_id)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        y_pred = (y_pred > 0.5).float()
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        if precision + recall == 0:
            return torch.tensor(0.0)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

def build_datasets(fold):
    print('[INFO] Preprando dataset landsat-sentinel')
    transform_train = RandomFlipRotate()
    transform = None 

    root_dir_map = {'landsat': LANDSAT_DATASET_PATH, 'sentinel': SENTINEL_DATASET_PATH, }
    mask_folder_map = {'landsat': LANDSAT_ANNOTATION_FOLDER, 'sentinel': SENTINEL_ANNOTATION_FOLDER,}
    img_folder_map = {'landsat': LANDSAT_IMG_FOLDER, 'sentinel': SENTINEL_IMG_FOLDER,}
    bands = {'landsat': LANDSAT_BANDS, 'sentinel': SENTINEL_BANDS,}
    
    quantification = {'landsat': LANDSAT_QUANTIFICATION, 'sentinel': SENTINEL_QUANTIFICATION}
    mean_std = None
   
    train_dataset = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=fold, set='train', transform=transform_train, means_stds=mean_std)
    val_dataset = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=fold, set='validation', transform=transform, means_stds=mean_std)


    test_dataset_full = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=fold, set='test', transform=transform, means_stds=mean_std)
    test_dataset_landsat = MultiSatelliteDataset(LANDSAT_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=fold, set='test', transform=transform, means_stds=mean_std)
    test_dataset_sentinel = MultiSatelliteDataset(SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=fold, set='test', transform=transform, means_stds=mean_std)


    return train_dataset, val_dataset, test_dataset_full, test_dataset_landsat, test_dataset_sentinel



def evaluate_dataset(trainer, module, test_dataset, batch_size=1, workers=4, keys=['backbone_prob']):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)
    module.eval()

    final_metrics = None
    if trainer is not None:
        print('[INFO] Iniciando avaliação...')
        final_metrics = trainer.test(module, test_dataloader)
        print('[INFO] Avaliação finalizada')
    
    print(f'[INFO] Computando métricas...')
    # cm_matrix = BinaryConfusionMatrix()    
    # test_precision = Precision(task="binary")
    # test_recall = Recall(task="binary")
    # test_f1 = F1Score(task="binary")

    precision = {}
    recall = {}
    f1 = {}
    cm = {}

    for key in keys:
        precision[key] = Precision(task="binary")
        recall[key] = Recall(task="binary")
        f1[key] = F1Score(task="binary")
        cm[key] = BinaryConfusionMatrix()

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images = batch['image'].to(module.device)
            masks_batch = (batch['mask'].to(module.device) > 0.5).float()
            # sensor_id = batch['satellite'].argmax(dim=1).to(module.device)  # Converte para o tipo correto
            # masks_batch = masks_batch.unsqueeze(1)  # Adiciona a dimensão de canal
            domain_id = batch['domain_id'].to(module.device)

            y_hat = module(images, domain_id=domain_id)
            # y_hat = torch.sigmoid(y_hat).squeeze(1)  # Remove a dimensão de canal
            # y_hat = torch.sigmoid(y_hat)
            for key in keys:
                probs = y_hat[key]
                preds = (probs > 0.5).float()

                precision[key].update(preds, masks_batch)
                recall[key].update(preds, masks_batch)
                f1[key].update(preds, masks_batch)
                cm[key].update(preds, masks_batch)

                # comparações entre branches
                for key2 in keys:
                    if key == key2:
                        continue

                    pair_key = f"{key}_vs_{key2}"
                    if pair_key not in precision:
                        precision[pair_key] = Precision(task="binary")
                        recall[pair_key] = Recall(task="binary")
                        f1[pair_key] = F1Score(task="binary")
                        cm[pair_key] = BinaryConfusionMatrix()
                    
                    preds2 = (y_hat[key2] > 0.5).float()
                    precision[pair_key].update(preds, preds2)
                    recall[pair_key].update(preds, preds2)
                    f1[pair_key].update(preds, preds2)
                    cm[pair_key].update(preds, preds2)



    precision = {k: v.compute() for k, v in precision.items()}
    recall = {k: v.compute() for k, v in recall.items()}
    f1 = {k: v.compute() for k, v in f1.items()}
    cm = {k: v.compute() for k, v in cm.items()}
    
    module.train()

    return precision, recall, f1, cm, final_metrics


def evaluate_numpy_rule(module, test_dataset):
    """
    Avalia regras numpy exportadas para todos os sensores presentes no test_dataset.
    Assume que cada sample contém ['image', 'mask', 'sensor_id'].
    """
    all_preds = []
    all_gts = []

    for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        sample = test_dataset[i]
        img = sample['image'].numpy().transpose(1,2,0)  # (H,W,3)
        gt = sample['mask'].numpy().squeeze(0)          # (H,W)

        gt = (gt > 0.5).astype(np.uint8)  # binariza ground truth

        sensor_id = int(sample['domain_id'])     # 0 = Sentinel, 1 = Landsat (por exemplo)

        # exporta regra específica para o sensor
        rule_fn = module.threshold_branch.export_numpy_rule(sensor_id)
        pred_mask = rule_fn(img)  # (H,W) float [0,1]

        # binariza (threshold fixo, pode calibrar depois)
        pred_bin = (pred_mask > 0.5).astype(np.uint8)

        all_preds.append(pred_bin.flatten())
        all_gts.append(gt.flatten())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    print("[DEBUG] Valores únicos em all_gts:", np.unique(all_gts), 'Shape:', all_gts.shape)
    print("[DEBUG] Valores únicos em all_preds:", np.unique(all_preds), 'Shape:', all_preds.shape)

    # Garante que os valores estão no intervalo esperado (0 ou 1)
    assert set(np.unique(all_gts)).issubset({0, 1}), "all_gts contém valores inesperados!"
    assert set(np.unique(all_preds)).issubset({0, 1}), "all_preds contém valores inesperados!"

    # Corrige valores inesperados em all_gts e all_preds
    all_gts = np.clip(all_gts, 0, 1)
    all_preds = np.clip(all_preds, 0, 1)

    precision = precision_score(all_gts, all_preds, zero_division=0)
    recall = recall_score(all_gts, all_preds, zero_division=0)
    f1 = f1_score(all_gts, all_preds, zero_division=0)
    cm = confusion_matrix(all_gts, all_preds)

    return precision, recall, f1, cm, None



def save_results_to_file(file_path, dataset_name, final_metrics, precision, recall, f1, cm_matrix):
    with open(file_path, 'a+') as f:
        f.write('-'*80 + '\n')
        f.write(f'Dataset: {dataset_name}\n')
        f.write(str(final_metrics))
        f.write(f'\nSklearn Metrics:\n')
        f.write(f'Precision: {str(precision)}\n')	
        f.write(f'Recall: {str(recall)}\n')
        f.write(f'F1: {str(f1)}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(str(cm_matrix))
        f.write('\n')


def create_spectralgpt(weights=None):
    model = vit_base_patch8_256_channel3(num_classes=1, seg_classes=1)
    if weights is not None:
        checkpoint = torch.load(weights, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % weights)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['pos_embed','patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        keys = list(checkpoint_model.keys())
        for k in keys:
            if k not in state_dict:
                print(f"Removing key {k} from pretrained checkpoint - Missing key")
                del checkpoint_model[k]
            elif checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint - Different weights size")
                del checkpoint_model[k]

        new_keys = list(checkpoint_model.keys())
        print('[INFO] Num. checkpoint keys:', len(keys), ' Num. new keys:', len(new_keys), ' Num. deleted keys:', len(keys)-len(new_keys))

        interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    return model

if __name__ == "__main__":
   
        
    parser = ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--devices', nargs='+', type=int, default=[1])

    args = parser.parse_args()

    fold = args.fold
    LR = args.lr
    EPOCHS = args.epochs
    ES_PATIENCE = args.es_patience
    WORKERS = args.workers
    DEVICES = args.devices

    log_dir = os.path.join(LOG_DIR, f'fold-{fold}')
    os.makedirs(log_dir, exist_ok=True)
    LOG_DIR = log_dir
   
    train_dataset, val_dataset, test_dataset_full, test_dataset_landsat, test_dataset_sentinel = build_datasets(fold)

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


    module = LitDTBranchAdvanced(backbone=BACKBONE, lr=LR)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(LOG_DIR, 'checkpoints'), save_top_k=1, monitor="val_loss", verbose=False, mode="min")


    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(default_root_dir=LOG_DIR, check_val_every_n_epoch=1, log_every_n_steps=1, accelerator=accelerator, max_epochs=EPOCHS, devices=DEVICES, callbacks=[early_stop_callback, checkpoint_callback], deterministic=True)
    
    print('[INFO] Iniciando treinamento...')
    

    if OVERRIDE_TRAINED or not os.path.exists(os.path.join(LOG_DIR, 'checkpoints', 'final_model.ckpt')):
        start_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        end_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('[INFO] Treinamento finalizado')
        
        trainer.save_checkpoint(os.path.join(LOG_DIR, 'checkpoints', 'last_trained_epoch.ckpt'))
        if checkpoint_callback.best_model_path:
            print(f'[INFO] Melhor checkpoint encontrado: {checkpoint_callback.best_model_path}')
            print('[INFO] Carregando melhor checkpoint...')
            module = LitDTBranchAdvanced.load_from_checkpoint(checkpoint_callback.best_model_path)
            print('[INFO] Melhor checkpoint carregado!')

            print('[INFO] Salvando melhor modelo como final_model.ckpt...')
            shutil.copy(checkpoint_callback.best_model_path, os.path.join(LOG_DIR, 'checkpoints', 'final_model.ckpt'))
            print('[INFO] Modelo final salvo (cópia do melhor checkpoint)!')
        
        else:
            print('[INFO] Salvando modelo...')
            trainer.save_checkpoint(os.path.join(LOG_DIR, 'checkpoints', 'final_model.ckpt'))
            print('[INFO] Modelo final salvo!')

    else:
        print('[INFO] Carregando modelo treinado...')
        module = LitDTBranchAdvanced.load_from_checkpoint(os.path.join(LOG_DIR, 'checkpoints', 'final_model.ckpt'))
        print('[INFO] Modelo treinado carregado!')
        trainer = None  # Não precisamos do trainer para avaliação  


    print(f'[INFO] Exportando regras numéricas...')
    rules = module.threshold_branch.export_rules_numeric()
    with open(os.path.join(LOG_DIR, 'exported_rules.json'), 'w') as f:
        json.dump(rules, f, indent=4)



    with open(os.path.join(LOG_DIR, 'final_results.txt'), 'w') as f:
        f.write(f'Resultados Finais - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Configurações:\n')
        f.write(f'Fold: {FOLD}\n')
        f.write(f'Backbone: {BACKBONE}\n')
        f.write(f'Learning Rate: {LR}\n')
        f.write(f'Batch Size: {BATCH_SIZE}\n')
        f.write(f'Epochs: {EPOCHS}\n')
        f.write(f'ES Patience: {ES_PATIENCE}\n')
        f.write(f'Devices: {DEVICES}\n')
        f.write(f'Override Trained: {OVERRIDE_TRAINED}\n')
        f.write('-'*80 + '\n')

    print(f'[INFO] Avaliando dataset Landsat-Sentinel...')  
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(trainer, module, test_dataset_full, batch_size=BATCH_SIZE, workers=WORKERS)
    print(f'[INFO] Salvando resultados Landsat-Sentinel...')
    save_results_to_file(os.path.join(LOG_DIR, 'final_results.txt'), 'Landsat-Sentinel', final_metrics, precision, recall, f1, cm_matrix)
        
    
    print(f'[INFO] Avaliando dataset Landsat...')
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(trainer, module, test_dataset_landsat, batch_size=BATCH_SIZE, workers=WORKERS)
    print(f'[INFO] Salvando resultados Landsat...')
    save_results_to_file(os.path.join(LOG_DIR, 'final_results.txt'), 'Landsat', final_metrics, precision, recall, f1, cm_matrix)

    print(f'[INFO] Avaliando dataset Sentinel...')
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(trainer, module, test_dataset_sentinel, batch_size=BATCH_SIZE, workers=WORKERS)
    print(f'[INFO] Salvando resultados - Sentinel...')
    save_results_to_file(os.path.join(LOG_DIR, 'final_results.txt'), 'Sentinel', final_metrics, precision, recall, f1, cm_matrix)

    print(f'[INFO] Processo finalizado!')


