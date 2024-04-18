# ===============
# Import
# ===============
import numpy as np
import pandas as pd
import torch
from torch import nn

# ===============
# import mycode
# ===============
import sys
sys.path.append('../')
#utils
from utils.utils import pickle_dump,pickle_load


# ==============================
# read_result
# ==============================
def softmax_second_axis(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def read_result(exp_id,col_labels,output_dir,use_tta=False,print_data=False,all_mode=False):
    load_output_dir = f'{output_dir}/{exp_id}'
    if print_data:
        print(load_output_dir)
    history         = pickle_load(f'{load_output_dir}/history.pkl')
    train_meta      = pd.read_csv(f'{load_output_dir}/train_meta.csv')
    Calc_Fold       = list(history.keys())
    Calc_Fold       = [x for x in Calc_Fold if isinstance(x, (int, float))]#数字のみ残す
    if len(Calc_Fold)==0:
        Calc_Fold       = list(history.keys())
        Calc_Fold.remove('oof_MAE')
        Calc_Fold.remove('oof_ACC')
    for idx,id_fold in enumerate(Calc_Fold):
        df_output,df_label      = get_result2df(history,id_fold,col_labels,all_mode=all_mode)
        if idx==0:
            df_oof_output       = df_output
            df_oof_labels       = df_label
        else:
            df_oof_output       = pd.concat([df_oof_output,df_output],axis=0).reset_index(drop=True)
            df_oof_labels       = pd.concat([df_oof_labels,df_label],axis=0).reset_index(drop=True)
    if use_tta:
        try:
            df_oof_output           = pd.read_csv(f'{load_output_dir}/df_tta_ens_output.csv')
            df_oof_labels           = pd.read_csv(f'{load_output_dir}/df_tta_ens_labels.csv')
        except:
            print('tta not found')
    
    df_oof_output            = df_oof_output.rename(columns={'label_ids':'label_id','eeg_ids':'eeg_id','spectrogram_ids':'spectrogram_id','patient_ids':'patient_id'})
    df_oof_labels            = df_oof_labels.rename(columns={'label_ids':'label_id','eeg_ids':'eeg_id','spectrogram_ids':'spectrogram_id','patient_ids':'patient_id'})
    return df_oof_output,df_oof_labels,train_meta

def get_result2df(history,id_fold,col_labels,all_mode=False):
    #output
    df_output           = history[id_fold]['best_df_valid_output']
    df_output['fold']   = id_fold
    col_labels_logits   = [s.split('_')[0]+'_logits' for s in col_labels]

    col_df_output       = list(df_output.columns)
    if ('flg1_vote' in col_df_output)&(all_mode==False):#logitsが全データ分なのでflg1のみ残す
        logits          = history[id_fold]['best_valid_logits']
        df_output_all   = history[id_fold]['best_df_valid_output_all']
        select_flg1     = df_output_all['flg1_vote']==True
        logits          = logits[select_flg1.values]
        df_output[col_labels_logits] =  logits
    else:
        df_output[col_labels_logits] =  history[id_fold]['best_valid_logits']
    #label
    df_label            = history[id_fold]['best_df_valid_labels']
    df_label['fold']    = id_fold
    return df_output,df_label

# ==============================
# preprocess_ensemble
# ==============================
import torch.nn.functional as F
def preprocess_ensemble(exp_ids,paths,col_labels_feat,weight_init,
                        master_label,col_labels,calc_init=False):
    oof_data                = []
    #single
    for path in paths:
        df_oof              = pd.read_csv(path)
        oof_data.append(df_oof[col_labels_feat].values)
    #matome
    weight_init             = np.array(weight_init)/np.sum(np.array(weight_init))
    oof_data                = np.stack(oof_data)
    label_data              = master_label[col_labels].values
    weight_init             = torch.tensor(weight_init).float()
    oof_data                = torch.tensor(oof_data).float()
    label_data              = torch.tensor(label_data).float()
    criterion               = torch.nn.KLDivLoss(reduction='batchmean')
    if calc_init:
        print('=====================================')
        print('num_model:',len(weight_init))
        for idx in range(len(weight_init)):
            log_probs          = F.log_softmax(oof_data[idx], dim=1)
            kl_div             = criterion(log_probs, label_data)
            print(idx,np.round(kl_div.item(),3),exp_ids[idx])

        weights            = torch.tensor([w / sum(weight_init) for w in weight_init])
        weighted_average   = torch.tensordot(weights, oof_data, dims=([0], [0]))
        log_probs          = F.log_softmax(weighted_average, dim=1)
        kl_div             = criterion(log_probs, label_data)
        print('init:',np.round(kl_div.item(),6))

    return weight_init,oof_data,label_data,criterion,df_oof

# ==============================
# optuna
# ==============================
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
def objective(trial, weight_init,oof_data,label_data,criterion):
    weights            = [trial.suggest_float(f'weight_{i}', 0, 1) for i in range(len(weight_init))]
    weights            = torch.tensor([w / sum(weights) for w in weights]).to('cuda')
    weighted_average   = torch.tensordot(weights, oof_data, dims=([0], [0]))
    log_probs          = F.log_softmax(weighted_average, dim=1)
    kl_div             = criterion(log_probs, label_data)
    return kl_div

def calc_optuna(exp_ids,weight_init,oof_data,label_data,criterion,n_trials=1000):
    study               = optuna.create_study(direction='minimize')  
    study.optimize(lambda trial: objective(trial, weight_init,oof_data.to('cuda'),label_data.to('cuda'),criterion.to('cuda')),
                                            n_trials=n_trials)
    best_weights        = list(study.best_params.values())
    df_optuna           = pd.DataFrame({'exp_id':exp_ids,'weights':best_weights})
    print(f'Best value: {study.best_value}')
    print(f'Best parameters: {study.best_params}')

    best_weights        = torch.tensor([w / sum(best_weights) for w in best_weights])
    oof_data_ens_logits = torch.tensordot(torch.tensor(best_weights), oof_data, dims=([0], [0]))
    oof_data_ens        = nn.functional.softmax(oof_data_ens_logits , dim=1).float().detach().cpu()
    return df_optuna,np.array(oof_data_ens_logits),np.array(oof_data_ens)
