"""
Cria os folds para teste treino e validação dos modelos.
O código verifica as imagens anotadas para verificar quais tem fogo e quais não tem
Para desativar a estratificação basta alterar a flag de configuração.
É possível reservar amostras para validação, se ativo, será reservado um conjunto do mesmo tamanho do conjunto de teste.
"""
import sys
import os
import pandas as pd
from glob import glob
import rasterio
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

SENTINEL_IMAGES_PATH = '/hybrid-fire-segmentation/dataset/Sentinel2/manual_annotated/bkp_256/imgs'
SENTINEL_ANNOTATIONS_PATH = '/hybrid-fire-segmentation/dataset/Sentinel2/manual_annotated/bkp_256/mask1'

LANDSAT_IMAGES_PATH = '/hybrid-fire-segmentation/dataset/Landsat/landsat_patches/'
LANDSAT_ANNOTATIONS_PATH = '/hybrid-fire-segmentation/dataset/Landsat/manual_annotations_patches/'

SATELLITE = 'landsat-sentinel' # 'sentinel', 'landsat', 'landsat-sentinel'


CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH = f'./dataframes/num_fire_and_ember_pixels_per_patch_{SATELLITE}_gdal.csv'
OVERRIDE_CSV_NUM_FIRE_PIXELS_PER_PATCH = True

NUM_FOLDS = 5
RANDOM_SEED = 1
STRATIFIED_FOLDS = False
GENERATE_VALIDATION_FOLD = True

CSV_FOLDS_BASE_PATH = f'./dataframes/folds/{SATELLITE}'


OUTPUT_CSV_NAME = f'{SATELLITE}_folds.csv'

def load_annotations_from_dir(annotations_path):
    annotations_images = glob(annotations_path + '/*.tif')
    if SATELLITE == 'landsat':
        annotations_images = [a for a in annotations_images if 'v1' in a]
        # annotations_images = [a for a in annotations_images if 'v2' in a]


    print(f'Path: {annotations_path} - Num. images: ', len(annotations_images))
    return annotations_images


def generate_csv():

    if SATELLITE == 'sentinel':
        annotations_paths = load_annotations_from_dir(SENTINEL_ANNOTATIONS_PATH)
    elif SATELLITE == 'landsat':
        annotations_paths = load_annotations_from_dir(LANDSAT_ANNOTATIONS_PATH)
    elif SATELLITE == 'landsat-sentinel':
        annotations_paths = load_annotations_from_dir(SENTINEL_ANNOTATIONS_PATH) + [a for a in load_annotations_from_dir(LANDSAT_ANNOTATIONS_PATH) if 'v1' in a]
    print('Num. images: ', len(annotations_paths))
    print('Counting fire pixels in images...')
    # sys.exit()
    data = []
    for annotation_path in tqdm(annotations_paths):
        with rasterio.open(annotation_path) as src:
            annotation = src.read(1)

        annotation_name = os.path.basename(annotation_path)
        if ('_20m_stack_maskf_' in annotation_name) or ('_20m_stack_maskmerge_' in annotation_name) or ('_b12b11b8a_' in annotation_name):
            satellite = 'sentinel'
        elif ('_v1_' in annotation_name) or ('_v2_' in annotation_name):
            satellite = 'landsat'
        else:
            raise ValueError('Unknown satellite for annotation: ', annotation_name)
        # image_name = annotation_name.replace('_20m_stack_maskf_', '_20m_stack_').replace('_v1_', '_').replace('_mask_', '_')
        image_name = annotation_name.replace('_20m_stack_maskf_s256_', '_20m_stack_s256_').replace('_20m_stack_maskmerge_', '_20m_stack_s256_').replace('_v1_', '_').replace('_v2_', '_').replace('_mask_', '_').replace('_mask.tif', '.tif')

        num_fire_pixels = annotation.sum()

        data.append({
            'satellite' : satellite,
            'annotation': 'manual',
            'image': image_name,
            'mask1': annotation_name,
            'num_fire_pixels': num_fire_pixels,
            'category': satellite + '_' + ('fire' if num_fire_pixels > 0 else 'no-fire'),
        })


    os.makedirs(os.path.dirname(CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH), exist_ok=True)
    df_patches = pd.DataFrame(data)
    df_patches.to_csv(CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH)

    return df_patches

def show_num_patches_in_each_category(df_patches):

    print('Num. patches with fire: ', len(df_patches[ df_patches['category'].str.endswith('fire') ]))
    print('Num. patches Annotations without fire: ', len(df_patches[ (df_patches['annotation'] == 'manual') & (df_patches['category'].str.endswith('no-fire')) ]))
    print('Num. patches Landsat-8: ', len(df_patches[ df_patches['satellite'] == 'landsat' ]))
    print('Num. patches Sentinel-2: ', len(df_patches[ df_patches['satellite'] == 'sentinel' ]))

    
    print('Num. Landsat-8 patches with fire: ', len(df_patches[ (df_patches['category'].str.endswith('fire')) & (df_patches['satellite'] == 'landsat') ]))
    print('Num. Landsat-8 patches Annotations without fire: ', len(df_patches[ (df_patches['annotation'] == 'manual') & (df_patches['category'].str.endswith('no-fire')) & (df_patches['satellite'] == 'landsat') ]))
    print('Num. Sentinel-2 patches with fire: ', len(df_patches[ (df_patches['category'].str.endswith('fire')) & (df_patches['satellite'] == 'sentinel') ]))
    print('Num. Sentinel-2 patches Annotations without fire: ', len(df_patches[ (df_patches['annotation'] == 'manual') & (df_patches['category'].str.endswith('no-fire')) & (df_patches['satellite'] == 'sentinel') ]))

 


if __name__ == '__main__':

    if OVERRIDE_CSV_NUM_FIRE_PIXELS_PER_PATCH or not os.path.exists(CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH):
        generate_csv()

    df_patches = pd.read_csv(CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH, index_col=0)

    satellites = SATELLITE.split('-')
    if len(satellites) == 1:
        df_patches = df_patches[ df_patches['satellite'] == SATELLITE ]
    else:
        df_patches = df_patches[ df_patches['satellite'].isin(satellites) ]

    show_num_patches_in_each_category(df_patches)

    X_patches = df_patches.drop('category', axis=1)
    y_categories = df_patches['category']
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    if STRATIFIED_FOLDS:
        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    os.makedirs(CSV_FOLDS_BASE_PATH, exist_ok=True)
    folds = []
    for _, test_index in kfold.split(X_patches, y_categories):
        folds.append(test_index)
    

    dataframes_folds = []
    for k, fold in enumerate(folds):
        print('Fold', k+1)

        df_val = pd.DataFrame(columns=X_patches.columns)
        if GENERATE_VALIDATION_FOLD:
            k_val = (k + 1) % len(folds)
            df_val = df_patches.iloc[folds[k_val]].copy()

        df_test = df_patches.iloc[fold].copy()
        df_train = df_patches[ (~df_patches.index.isin(df_test.index)) & (~df_patches.index.isin(df_val.index) )].copy()
        
        print('Fold: ', k+1)
        print(len(df_train), len(df_val), len(df_test))
        print('Train:')
        show_num_patches_in_each_category(df_train)
        print('\n\nValidation:')
        show_num_patches_in_each_category(df_val)
        print('\n\nTest')
        show_num_patches_in_each_category(df_test)
        print('-'*80)
        
        df_train['set'] = 'train'
        df_val['set'] = 'validation'
        df_test['set'] = 'test'

        df_fold = pd.concat((df_train, df_val, df_test))
        df_fold.reset_index(inplace=True, drop=True)
        df_fold['fold'] = k+1
        
        df_fold.to_csv(os.path.join(CSV_FOLDS_BASE_PATH, f'{SATELLITE}_fold_{k+1}.csv'))

        dataframes_folds.append(df_fold.copy())

    df_all_folds = pd.concat(dataframes_folds)
    print(df_all_folds.value_counts('fold'))

    df_all_folds.to_csv(os.path.join(CSV_FOLDS_BASE_PATH, OUTPUT_CSV_NAME))
    print('Done!')
