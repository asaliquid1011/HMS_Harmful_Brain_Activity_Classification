# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold

# ===============
# import mycode  
# ===============
import sys
sys.path.append('../')
from utils.utils import pickle_dump,pickle_load,seed_everything,AttrDict,replace_placeholders
from utils.logger import setup_logger, LOGGER

def split_data(df_train,config):
    output_dir              = config['path']['output_dir']
    SplitTYPE               = config['calc_mode']['Split_Mode']
    N_FOLDS                 = config['calc_mode']['NFOLDS']
    SEED                    = config['train']['SEED']
    df_train['fold']  = -1
    if SplitTYPE == 'StratifiedKFold':
        for n_fold,(train_index, test_index) in enumerate(StratifiedKFold(n_splits=N_FOLDS).split(df_train, y=df_train['patient_id'])):
            df_train['fold'][test_index]  = n_fold
            print(len(df_train[df_train['fold']==n_fold]['patient_id'].unique()))
    elif SplitTYPE == 'GroupKFold':
        num_samples   = []
        num_patients  = []
        num_labels    = []
        for n_fold,(train_index, test_index) in enumerate(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(df_train)), 
                                                                                            y=df_train['expert_consensus'].values, 
                                                                                            groups=df_train['patient_id'].values)):
            df_train['fold'][test_index] = n_fold
            num_samples.append(len(test_index))
            num_patients.append(len(df_train[df_train['fold']==n_fold]['patient_id'].unique()))

            #patient_id_eeg_id毎のexpert_consensus
            tmp         = df_train[df_train['fold']==n_fold].groupby(['patient_id', 'eeg_id']).first().reset_index()
            num_labels.append(tmp['expert_consensus'].value_counts())
        print('num_samples: ',num_samples)
        print('num_patients:',num_patients)
        print('num_labels:',num_labels)
        file_splits         = list(df_train['fold'].values)
        pickle_dump(file_splits,f'{output_dir}/file_splits_def.pkl')
    elif SplitTYPE == 'def':
        file_splits         = pickle_load(f'{output_dir}/file_splits_def.pkl')
        df_train['fold']    = file_splits
        
    return df_train


def resplit_data(train_meta,config):
    num_swap                        = config['calc_mode']['num_swap']
    seed                            = config['train']['SEED']
    print('====Re Split_Data====')
    patient_fold_dict               = train_meta[['patient_id', 'fold']].drop_duplicates().set_index('patient_id')['fold'].to_dict()
    patient_ids                     = list(patient_fold_dict.keys())
    for id_num in range(num_swap):
        np.random.seed(seed+2*id_num)
        selected_patients           = np.random.choice(patient_ids, size=2, replace=False)
        patient_fold_dict[selected_patients[0]], patient_fold_dict[selected_patients[1]] = patient_fold_dict[selected_patients[1]], patient_fold_dict[selected_patients[0]]
    # Update
    train_meta['fold']              = train_meta['patient_id'].map(patient_fold_dict)

    #分布確認
    num_samples   = []
    num_patients  = []
    num_labels    = []
    for id_fold in config.calc_mode.Calc_Fold:
        valid_meta_fold              = train_meta[train_meta['fold']==id_fold].reset_index(drop=True)
        num_samples.append(len(valid_meta_fold))
        num_patients.append(len(valid_meta_fold['patient_id'].unique()))
        #patient_id_eeg_id毎のexpert_consensus
        tmp                          = valid_meta_fold[valid_meta_fold['fold']==id_fold].groupby(['patient_id', 'eeg_id']).first().reset_index()
        num_labels.append(tmp['expert_consensus'].value_counts())
    print('num_samples: ',num_samples)
    print('num_patients:',num_patients)
    print('num_labels:',num_labels)

    return train_meta
