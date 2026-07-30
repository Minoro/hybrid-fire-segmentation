import os 
import argparse
from datetime import datetime

import torch


from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lightning.pytorch import Trainer, seed_everything

from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix


from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


from my_dataset import MultiSatelliteDataset, calculate_mean_std

from util.pos_embed import interpolate_pos_embed
from src.models_vit_tensor_CD_2 import vit_base_patch8_256_channel3

import pandas as pd

seed_everything(42)

LANDSAT_DATASET_PATH = '/dataset/Landsat/GROUNDTRUTH'
LANDSAT_ANNOTATION_FOLDER = 'GROUNDTRUTH_GABRIEL_patches_cp'
LANDSAT_IMG_FOLDER = 'IMG_all_patches'
LANDSAT_BANDS = (7,6,5)

SENTINEL_DATASET_PATH = '/dataset/ActiveFire256/'
SENTINEL_ANNOTATION_FOLDER = 'annotations'
SENTINEL_IMG_FOLDER = 'imgs_256_with_nodata'
SENTINEL_BANDS = (6,5,4)

MODIS_DATASET_PATH = '/dataset/modis'
MODIS_ANNOTATIONS_PATH = '/dataset/modis'
MODIS_ANNOTATION_FOLDER = 'masks'
MODIS_IMG_FOLDER = 'images'
MODIS_BANDS = (7,6,2)

LANDSAT_DATAFRAME_PATH = f'/spectralgpt/active_fire/dataframes/samples_256_landsat_8020.csv'
SENTINEL_DATAFRAME_PATH = '/spectralgpt/active_fire/dataframes/samples_256_sentinel_8020.csv'
LANDSAT_SENTINEL_DATAFRAME_PATH = '/spectralgpt/active_fire/dataframes/folds/landsat-sentinel_folds.csv'
LANDSAT_SENTINEL_MODIS_DATAFRAME_PATH = '/spectralgpt/active_fire/dataframes/folds/landsat-sentinel-modis_all_folds.csv'


LOG_DIR = '/spectralgpt/downstream_tasks/spectralgpt/logs-multisatellite'
# TENSOR_BOARD_DIR = '/spectralgpt/downstream_tasks/unet/logs/tensorboard'
# CHECKPOINT_DIR = '/spectralgpt/downstream_tasks/unet/logs/checkpoints'






class SegNetModule(L.LightningModule):
    def __init__(self, model='spectralgpt', encoder_name='default', num_classes=1, in_channels=3, weights=None, lr=1e-3, freeze_encoder=False):
        super().__init__()

        # weights = ResNet50_Weights.LANDSAT_OLI_SR_MOCO
        self.model_name = model
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.weights = weights
        self.lr = lr
        self.save_hyperparameters()

        self.train_confusion_matrix = BinaryConfusionMatrix()
        self.val_confusion_matrix = BinaryConfusionMatrix()
        self.test_confusion_matrix = BinaryConfusionMatrix()


        self.train_metrics = torchmetrics.MetricCollection(
            {
                "f1": torchmetrics.classification.BinaryF1Score(),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model = vit_base_patch8_256_channel3(num_classes=self.num_classes, seg_classes=1)
        if weights is not None:
            checkpoint = torch.load(weights, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % weights)
            checkpoint_model = checkpoint['model']
            state_dict = self.model.state_dict()
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

            interpolate_pos_embed(self.model, checkpoint_model)

            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

        

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        # print(x)
        y_hat = self.model(x)
        
        y_hat = y_hat['out'].squeeze(1)
        y = y.float()

        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        batch_value = self.train_metrics(nn.functional.sigmoid(y_hat), y)
        self.log_dict(batch_value, sync_dist=True)

        self.train_confusion_matrix(y_hat.flatten(), y.flatten())


        return {'loss': loss}

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        y_hat = self.model(x)
        
        y_hat = y_hat['out'].squeeze(1)
        y = y.float()
        
        loss = self.loss_fn(y_hat, y)

        
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        batch_value = self.valid_metrics(nn.functional.sigmoid(y_hat), y)
        self.log_dict(batch_value, sync_dist=True)
        self.val_confusion_matrix(y_hat.flatten(), y.flatten())

        return {'loss' : loss}

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self.model(x)
        
        y_hat = y_hat['out'].squeeze(1)
        y = y.float()
        
        loss = self.loss_fn(y_hat, y)

        self.log('test_loss', loss, sync_dist=True)

        batch_value = self.test_metrics(nn.functional.sigmoid(y_hat), y, prog_bar=True)
        self.log_dict(batch_value, sync_dist=True)
        self.test_confusion_matrix(y_hat.flatten(), y.flatten())
        
        return {'loss': loss}

    def configure_optimizers(self):
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_optimize, lr=self.lr)
        return optimizer




def build_datasets(args):
    print(f'[INFO] Preprando dataset {args.satellite}...')
    transform = None

    root_dir_map = {'landsat': LANDSAT_DATASET_PATH, 'sentinel': SENTINEL_DATASET_PATH, 'modis': MODIS_DATASET_PATH}
    mask_folder_map = {'landsat': LANDSAT_ANNOTATION_FOLDER, 'sentinel': SENTINEL_ANNOTATION_FOLDER, 'modis': MODIS_ANNOTATION_FOLDER}
    img_folder_map = {'landsat': LANDSAT_IMG_FOLDER, 'sentinel': SENTINEL_IMG_FOLDER, 'modis': MODIS_IMG_FOLDER}
    bands = {'landsat': LANDSAT_BANDS, 'sentinel': SENTINEL_BANDS, 'modis': MODIS_BANDS}
    
    quantification = {'landsat': args.quantification, 'sentinel': args.quantification, 'modis': args.quantification}
    if isinstance(args.quantification, list):
        print(args.quantification)
        if len(args.quantification) == 1:
            quantification = {args.satellite: args.quantification[0]}
        elif len(args.quantification) == 2:
            quantification = {'landsat': args.quantification[0], 'sentinel': args.quantification[1]}
        elif len(args.quantification) == 3:
            quantification = {'landsat': args.quantification[0], 'sentinel': args.quantification[1], 'modis': args.quantification[2]}

    mean_std = None
    if 'mean-std' in args.quantification:
        mean_std = define_means_stds(args)

    train_dataset = MultiSatelliteDataset(args.dataframe_path, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='train', transform=transform, means_stds=mean_std)
    val_dataset = MultiSatelliteDataset(args.dataframe_path, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='validation', transform=transform, means_stds=mean_std)
    test_dataset = MultiSatelliteDataset(args.dataframe_path, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=args.fold, set='test', transform=transform, means_stds=mean_std)
    
    return train_dataset, val_dataset, test_dataset


def define_means_stds(args):
    if type(args.quantification) != list:
        return None
    
    df_folds = pd.read_csv(args.dataframe_path)
    df_folds = df_folds[ (df_folds['fold'] == args.fold) & (df_folds['set'] == 'train') ]

    root_dir_map = {'landsat': LANDSAT_DATASET_PATH, 'sentinel': SENTINEL_DATASET_PATH, 'modis': MODIS_DATASET_PATH}
    mask_folder_map = {'landsat': LANDSAT_ANNOTATION_FOLDER, 'sentinel': SENTINEL_ANNOTATION_FOLDER, 'modis': MODIS_ANNOTATION_FOLDER}
    img_folder_map = {'landsat': LANDSAT_IMG_FOLDER, 'sentinel': SENTINEL_IMG_FOLDER, 'modis': MODIS_IMG_FOLDER}
    bands = {'landsat': LANDSAT_BANDS, 'sentinel': SENTINEL_BANDS, 'modis': MODIS_BANDS}


    df_folds['root_dir'] = df_folds['satellite'].apply(lambda x : root_dir_map[x])
    df_folds['img_folder'] = df_folds['satellite'].apply(lambda x : img_folder_map[x])
    df_folds['mask_folder'] = df_folds['satellite'].apply(lambda x : mask_folder_map[x])

    df_folds['image_path'] = df_folds.apply(lambda x : os.path.join(x['root_dir'], x['img_folder'], x['image']), axis=1)
    df_folds['mask_path'] = df_folds.apply(lambda x : os.path.join(x['root_dir'], x['mask_folder'], x['mask1']), axis=1)

    landsat_mean_std = None
    sentinel_mean_std = None
    modis_mean_std = None
    if len(args.quantification) == 1 and args.quantification[0] == 'mean-std':
        return {args.satellite: calculate_mean_std(df_folds[ df_folds['satellite'] == args.satellite], 'image_path', bands[args.satellite])}

    if len(args.quantification) >= 1 and args.quantification[0] == 'mean-std':
        landsat_mean_std = calculate_mean_std(df_folds[ df_folds['satellite'] == 'landsat' ], 'image_path', LANDSAT_BANDS)
    
    if len(args.quantification) >= 2 and args.quantification[1] == 'mean-std':
        sentinel_mean_std = calculate_mean_std(df_folds[ df_folds['satellite'] == 'sentinel' ], 'image_path', SENTINEL_BANDS)
    
    if len(args.quantification) == 3 and args.quantification[2] == 'mean-std':
        modis_mean_std = calculate_mean_std(df_folds[ df_folds['satellite'] == 'modis' ], 'image_path', MODIS_BANDS)

    

    if args.satellite == 'landsat':
        return {'landsat': landsat_mean_std}
    elif args.satellite == 'sentinel':
        return {'sentinel': sentinel_mean_std}
    elif args.satellite == 'landsat-sentinel':
        return {'landsat': landsat_mean_std, 'sentinel': sentinel_mean_std}
    elif args.satellite == 'landsat-sentinel-modis':
        return {'landsat': landsat_mean_std, 'sentinel': sentinel_mean_std, 'modis': modis_mean_std}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--satellite', choices=['sentinel', 'landsat', 'landsat-sentinel', 'landsat-sentinel-modis'], type=str, required=True)
    parser.add_argument('--quantification', default=1.0, nargs='+')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--dataframe-path', default=LANDSAT_SENTINEL_DATAFRAME_PATH, type=str)

    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='Máx number of total epochs to run')
    parser.add_argument('-p', '--early-stopping-patience', default=10, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-d', '--devices', default=1, type=int, help='Number of devices to use')
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
    elif args.satellite == 'modis':
        bands_id = ''.join(list(map(str, MODIS_BANDS)))
    elif args.satellite == 'landsat-sentinel':
        bands_id = ''.join(list(map(str, LANDSAT_BANDS + SENTINEL_BANDS)))
    elif args.satellite == 'landsat-sentinel-modis':
        bands_id = ''.join(list(map(str, LANDSAT_BANDS + SENTINEL_BANDS + MODIS_BANDS)))

    quantification_id = ''.join(list(map(str, list(args.quantification))))

    if args.freeze_encoder:
        log_dir = os.path.join(args.log_dir, args.satellite, f'{args.model}-{args.encoder}-{weights_id}-b{bands_id}-q{quantification_id}-e{args.epochs}-freeze-encoder', str(args.fold))
    else:
        log_dir = os.path.join(args.log_dir, args.satellite, f'{args.model}-{args.encoder}-{weights_id}-b{bands_id}-q{quantification_id}-e{args.epochs}', str(args.fold))
    
    os.makedirs(log_dir, exist_ok=True)


    start_at = datetime.now()

    with open(os.path.join(log_dir, 'args.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write(f'Started at: {start_at}\n')
        f.write(str(args))
        f.write('\n')

    print(f'[INFO] Preprando dataset {args.satellite}...')
    transform = None

    train_dataset, val_dataset, test_dataset = build_datasets(args)

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

    in_channels = max(len(LANDSAT_BANDS), len(SENTINEL_BANDS), len(MODIS_BANDS))
    weights = None
    if args.weights is not None:
        if type(args.weights) == str and os.path.exists(args.weights):
            weights = args.weights
        # else:
        #     if args.encoder == 'resnet18':
        #         weights = ResNet18_Weights[args.weights]
        #     if args.encoder == 'resnet50':
        #         weights = ResNet50_Weights[args.weights]

        #     in_channels = weights.meta['in_chans']
    
    print(args.weights)
    # print(weights)
    
    module = SegNetModule(model=args.model, encoder_name=args.encoder, num_classes=args.num_classes, in_channels=in_channels, weights=weights, lr=args.lr, freeze_encoder=args.freeze_encoder)

        
    logger = TensorBoardLogger(save_dir=os.path.join(log_dir, 'tensorboard'), name='lightning_logs')

    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=int(args.early_stopping_patience), verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(log_dir, 'checkpoints'), save_top_k=1, monitor="valid_loss", verbose=False, mode="min")


    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(default_root_dir=log_dir, check_val_every_n_epoch=1, log_every_n_steps=1, accelerator=accelerator, max_epochs=args.epochs, devices=[args.devices], logger=logger, callbacks=[early_stop_callback, checkpoint_callback], deterministic=False)

    print('[INFO] Iniciando treinamento...')
    trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print('[INFO] Treinamento finalizado')

    print('[INFO] Salvando modelo...')
    trainer.save_checkpoint(os.path.join(log_dir, 'checkpoints', 'final_model.ckpt'))
    print('[INFO] Modelo final salvo!')

    print('[INFO] Iniciando avaliação...')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    final_metrics = trainer.test(module, test_dataloader)
    print('[INFO] Avaliação finalizada')

    end_at = datetime.now()
    with open(os.path.join(log_dir, 'final_results.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write(f'Started at: {start_at}\n')
        f.write(f'Final results at: {end_at}\n')
        f.write(str(final_metrics))
        f.write('\n')

        
        
    
    print(f'[INFO] Computando métricas via sklearn...')
    preds = []
    masks = []
    for batch in tqdm(test_dataloader):
        y_hat = module.model(batch[0])['out']
        y_hat = y_hat.float()
        y_hat = nn.functional.sigmoid(y_hat)
        y_hat = y_hat.squeeze(1)
        
        y_hat = y_hat > 0.5
        
        preds.append(y_hat)
        masks.append(batch[1])

    
    ypred = torch.cat(preds).detach().cpu().numpy()
    ytarget = torch.cat(masks).detach().cpu().numpy()


    ypred = ypred.flatten()
    ytarget = ytarget.flatten()

    precision = precision_score(ytarget, ypred)
    recall = recall_score(ytarget, ypred)
    f1 = f1_score(ytarget, ypred)
    report = classification_report(ytarget, ypred)
    cm_matrix = confusion_matrix(ytarget, ypred)

    print('Precision:',precision)
    print('Recall:', recall)
    print('F1:', f1)
    print(str(report))
    print(str(cm_matrix))

    print(f'[INFO] Salvando resultados...')
    with open(os.path.join(log_dir, 'final_results.txt'), 'a+') as f:
        f.write('-'*80 + '\n')
        f.write('Args:\n')
        f.write(str(args))
        f.write('\n')
        f.write(f'Sklearn Metrics:\n')
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

    


