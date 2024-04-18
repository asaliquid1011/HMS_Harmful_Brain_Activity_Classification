# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import os
import numpy as np
import pandas as pd
import random
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
# from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from matplotlib import pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import importlib

# ===============
# import mycode
# ===============
import sys
sys.path.append('../')
from trainer.metrics  import score

import trainer.datasets_pp
import models.model_HMS_PP
importlib.reload(trainer.datasets_pp)
importlib.reload(models.model_HMS_PP)
from trainer.datasets_pp import HMS_Dataset_PP
from models.model_HMS_PP import HMSModel_Stacking,HMSModel_PP

#utils
from utils.utils import pickle_dump,pickle_load,seed_everything,AttrDict,replace_placeholders
from utils.logger import setup_logger, LOGGER

# ===========================================================================
# train_loop
# ===========================================================================
def train_loop_pp(df_feat,col_feat,col_label,config,
                  save_output_dir,LOGGER,
                  mode='stacking'):
    
    folds                       = config.calc_mode.Calc_Fold
    DEVICE                      = 'cuda'
    BATCH_SIZE_Tr               = config['train']['BATCH_SIZE_Tr']
    BATCH_SIZE_Val              = config['train']['BATCH_SIZE_Val']
    NUM_WORKERS                 = config['train']['NUM_WORKERS']
    EPOCHS                      = config['train']['EPOCHS']
    LEARNING_RATE               = config['train']['LEARNING_RATE']
    WEIGHT_DECAY                = config['train']['WEIGHT_DECAY']

    #=Main=#
    history                     = defaultdict(list) 
    for id_fold in folds:
        LOGGER.info(f'============Fold:{id_fold} Start============')
        save_output_dir_fold    = f'{save_output_dir}/fold_{id_fold}'
        os.makedirs(save_output_dir_fold,exist_ok=True)
        #===split_data===#
        idx_train               = df_feat[df_feat['fold']!=id_fold].index.values
        idx_valid               = df_feat[df_feat['fold']==id_fold].index.values
        df_fold_train           = df_feat.iloc[idx_train].reset_index(drop=True)
        df_fold_valid           = df_feat.iloc[idx_valid].reset_index(drop=True)
        #===dataset===#
        train_dataset           = HMS_Dataset_PP(df_fold_train,col_feat,col_label,phase='train')
        valid_dataset           = HMS_Dataset_PP(df_fold_valid,col_feat,col_label,phase='valid')    
        train_loader            = DataLoader(train_dataset, batch_size=BATCH_SIZE_Tr, shuffle=True,
                                            num_workers=NUM_WORKERS,pin_memory=True)
        valid_loader            = DataLoader(valid_dataset, batch_size=BATCH_SIZE_Val, shuffle=False,
                                            num_workers=NUM_WORKERS,pin_memory=True)
        #===model===#
        if mode=='stacking':
            model               = HMSModel_Stacking(col_feat,config)
        elif mode=='pp':
            model               = HMSModel_PP(col_feat,config)
        model                   = model.to(DEVICE)

        #===loss===#
        loss_function           = torch.nn.KLDivLoss(reduction="batchmean")
        loss_function           = loss_function.to(DEVICE)

        #===optimizerl===#
        optimizer               = torch.optim.Adam(model.parameters(),
                                                lr=LEARNING_RATE ,
                                                weight_decay=WEIGHT_DECAY)
        #===schedulerl===#
        max_steps               = EPOCHS * len(train_loader)
        scheduler               = get_cosine_schedule_with_warmup(optimizer, num_training_steps=max_steps,
                                                                    num_warmup_steps=0)
        history_fold            = dict()
        history_fold            = calc_train_pp(model, config, train_loader, valid_loader,
                                                optimizer,scheduler,loss_function,save_output_dir_fold,LOGGER,
                                                EPOCHS,DEVICE,
                                                history_fold)
        
        # ===============
        # result_fold
        # =============== 
        LOGGER.info(f'============Fold:{id_fold} END============')
        history[id_fold]        = history_fold
        #history_fold['best_df_valid_output'].to_csv(f'{save_output_dir_fold}/df_valid_output.csv')
        #history_fold['best_df_valid_labels'].to_csv(f'{save_output_dir_fold}/df_valid_labels.csv')
        LOGGER.info('Best_Ep ' + str(history_fold['best_epoch']) +  ',Best_val_loss: ' + str(np.round(history_fold['best_valid_losses'],4))+
                    ',Best_val_metrics: ' + str(np.round(history_fold['best_valid_metrics'],4)))
        formatted_mae = [f'{val:.2f}' for val in history_fold['best_valid_MAE']]
        formatted_acc = [f'{val:.1f}' for val in history_fold['best_valid_ACC']]
        LOGGER.info(f'======Best_MAE : {formatted_mae}======')
        LOGGER.info(f'======Best_ACC : {formatted_acc}======')
        #folds
        history_fold['idx_train'] = idx_train
        history_fold['idx_valid'] = idx_valid

        # ===============
        # OOF_fold
        # =============== 
        LOGGER.info(f'============Fold:{id_fold} END============')
        del train_dataset,valid_dataset,train_loader,valid_loader
        gc.collect()
        torch.cuda.empty_cache()  

    # ===============
    # result_all
    # =============== 
    #===result_all===#
    best_epochs             = []
    folds                   = []
    best_train_losses       = []
    best_valid_losses       = []
    best_valid_metrics      = []
    best_valid_maes         = []
    best_valid_accs         = []
    for idx,id_fold in enumerate(config.calc_mode.Calc_Fold):
        folds.append(int(id_fold))
        best_epochs.append(int(history[id_fold]['best_epoch']))
        best_train_losses.append(history[id_fold]['best_train_losses'])
        best_valid_losses.append(history[id_fold]['best_valid_losses'])
        best_valid_metrics.append(history[id_fold]['best_valid_metrics'])
        best_valid_maes.append(history[id_fold]['best_valid_MAE'])
        best_valid_accs.append(history[id_fold]['best_valid_ACC'])
    df_result                               = pd.DataFrame()
    df_result['folds']                      = folds
    df_result['best_epochs']                = best_epochs
    df_result['best_train_losse']           = best_train_losses
    df_result['best_valid_losse']           = best_valid_losses
    df_result['best_valid_metrics']         = best_valid_metrics
    df_result                               = df_result.T

    #=df_oof=#
    for idx,id_fold in enumerate(config.calc_mode.Calc_Fold):
        if idx==0:
            df_oof_output                   = history[id_fold]['best_df_valid_output']
            df_oof_labels                   = history[id_fold]['best_df_valid_labels']
            df_oof_output['fold']           = id_fold
            df_oof_labels['fold']           = id_fold
        else:
            df_oof_output_tmp               = history[id_fold]['best_df_valid_output']
            df_oof_labels_tmp               = history[id_fold]['best_df_valid_labels']
            df_oof_output_tmp['fold']       = id_fold
            df_oof_labels_tmp['fold']       = id_fold
            df_oof_output                   = pd.concat([df_oof_output,df_oof_output_tmp],axis=0).reset_index(drop=True)
            df_oof_labels                   = pd.concat([df_oof_labels,df_oof_labels_tmp],axis=0).reset_index(drop=True)

    df_oof_output_score                     = df_oof_output[config['dataset']['col_labels']]
    df_oof_labels_score                     = df_oof_labels[config['dataset']['col_labels']]
    df_oof_output_score['id']               = np.arange(len(df_oof_output_score))
    df_oof_labels_score['id']               = np.arange(len(df_oof_labels_score))
    oof_metrics                             = score(solution=df_oof_labels_score.copy(), submission=df_oof_output_score.copy(), 
                                                    row_id_column_name='id')
    
    LOGGER.info('Best_Valid_Metrics_OOF    : '+str(np.round(oof_metrics,4)))
    LOGGER.info('Best_Valid_Metrics_FoldAve: '+str(np.round(np.mean(best_valid_metrics),4)))

    #===other===#
    proba_pred                              = torch.tensor(df_oof_output[config['dataset']['col_labels']].values)
    label                                   = torch.tensor(df_oof_labels[config['dataset']['col_labels']].values)
    #MAE
    mae                                     = abs(proba_pred-label).mean(axis=0).tolist()
    #acc
    acc                                     = []
    predicted_classes                       = torch.argmax(proba_pred, dim=1)
    label_classes                           = torch.argmax(label, dim=1)
    for c in range(proba_pred.shape[1]):
        class_mask                          = label_classes == c
        class_predictions                   = predicted_classes[class_mask]
        class_labels                        = label_classes[class_mask]
        correct_predictions                 = torch.eq(class_predictions, class_labels)
        accuracy                            = torch.mean(correct_predictions.float())
        acc.append(accuracy.item() * 100)
    history['oof_MAE']          = mae
    history['oof_ACC']          = acc
    formatted_mae = [f'{val:.2f}' for val in mae]
    formatted_acc = [f'{val:.1f}' for val in acc]
    LOGGER.info(f'======oof_MAE : {formatted_mae}======')
    LOGGER.info(f'======oof_ACC : {formatted_acc}======')

    return df_oof_output,oof_metrics


# ===========================================================================
# Model_Train
# ===========================================================================
def calc_train_pp(model, config,train_loader, val_loader,
                    optimizer,scheduler,loss_function,save_output_dir,LOGGER,
                    n_epochs,DEVICE,
                    history_fold,):
    #=logger=#
    LOGGER.info('===Starting training loop===')
    LOGGER.info("EPOCHS={}".format(n_epochs))
    #=initialize1:metric=#
    best_corrects        = 99999
    # ===============
    # Train loop
    # ===============
    train_losses_epochs = []
    valid_losses_epochs = []
    valid_metrics_epochs= []
    for e in range(0, n_epochs):     
        # ===============
        # Train
        # ===============
        train_losses                = []
        model.train()
        for batch, data in enumerate(train_loader):
            patient_id              = data['patient_id']
            eeg_id                  = data['eeg_id']
            spectrogram_id          = data['spectrogram_id']
            label                   = data['label']#label [4, 5760, 3]
            #input
            feat                    = data['feat']
            if torch.cuda.is_available():
                feat                = feat.float().to(DEVICE)
                label               = label.to(DEVICE)
            #=gradient_zero_clear=
            optimizer.zero_grad() 
            #=preds=
            (_,logits)              = model(feat)
            pred                    = nn.functional.log_softmax(logits, dim=1)
            #=loss=
            loss                    = loss_function(pred, label).mean()
            #=backpropagation=
            loss.backward()         
            optimizer.step()        
            #=get_loss_corr_metric_in_batch=
            with torch.no_grad():
                if batch==0:
                    train_losses        = [loss.float().detach().cpu()]
                else:
                    train_losses.append(loss.float().detach().cpu())
        # ===============
        # scheduler
        # =============== 
        scheduler.step()
        # ==============================
        # get_loss_corr_metric_all_batch
        # ============================== 
        with torch.no_grad():
            train_losses_mean           = np.mean(np.array(train_losses))  
        # ===============
        # Valid
        # ===============     
        with torch.no_grad():
            #=init=        
            model.eval()
            #=batch_loop=#
            for batch, data in enumerate(val_loader):
                patient_id              = data['patient_id']
                eeg_id                  = data['eeg_id']
                spectrogram_id          = data['spectrogram_id']
                label                   = data['label']#label [4, 5760, 3]
                #input
                feat                    = data['feat']
                if torch.cuda.is_available():
                    feat                = feat.float().to(DEVICE)
                    label               = label.to(DEVICE)
                #=preds=
                (_,logits)              = model(feat)
                pred                    = nn.functional.log_softmax(logits, dim=1)                  
                #=loss=
                loss                        = loss_function(pred, label).mean()
                if batch==0:
                    valid_patient_ids       = patient_id.tolist()
                    valid_eeg_ids           = eeg_id.tolist()
                    valid_spectrogram_ids   = spectrogram_id.tolist()
                    valid_logits            = logits.float().detach().cpu()
                    valid_output            = nn.functional.softmax(logits, dim=1).float().detach().cpu()
                    valid_labels            = label.float().detach().cpu()
                    valid_losses            = [loss.float().detach().cpu()]
                else:
                    valid_patient_ids       += patient_id.tolist()
                    valid_eeg_ids           += eeg_id.tolist()
                    valid_spectrogram_ids   += spectrogram_id.tolist()
                    valid_logits            = torch.cat([valid_logits,logits.float().detach().cpu()],dim=0)
                    valid_output            = torch.cat([valid_output,nn.functional.softmax(logits, dim=1).float().detach().cpu()],dim=0)
                    valid_labels            = torch.cat([valid_labels,label.float().detach().cpu()],dim=0)
                    valid_losses.append(loss.float().detach().cpu())
            # ===============
            # Valid Score
            # ===============  
            valid_losses_mean               = np.mean(np.array(valid_losses))
            # print(valid_losses)
            valid_logits                    = valid_logits.numpy()
            valid_output                    = valid_output.numpy()
            valid_labels                    = valid_labels.numpy()
            #score
            df_valid_output                 = pd.DataFrame(valid_output,columns=config['dataset']['col_labels'])
            df_valid_labels                 = pd.DataFrame(valid_labels,columns=config['dataset']['col_labels'])
            df_valid_output['id']           = np.arange(len(df_valid_output))
            df_valid_labels['id']           = np.arange(len(df_valid_labels))
            valid_metrics                   = score(solution=df_valid_labels.copy(), submission=df_valid_output.copy(), 
                                                    row_id_column_name='id')
            train_losses_epochs.append(train_losses_mean)
            valid_losses_epochs.append(valid_losses_mean)
            valid_metrics_epochs.append(valid_metrics)
            # =============================
            # Save model_best  Valid
            # =============== ==============
            if valid_metrics <= best_corrects:
                best_corrects                = valid_metrics
                try:
                    torch.save(model, save_output_dir+'/best.pt')
                except:#custom_classの場合
                    torch.save(model.state_dict(), save_output_dir+'/best_state_dict.pt')
                LOGGER.info('Ep ' + str(e) + ', ==Train END==, Train_Loss: ' + str(np.round(train_losses_mean,4)))
                LOGGER.info('======Saving new best model at epoch ' + str(e) +
                    ',  Val_Loss: ' + str(np.round(valid_losses_mean,4))+', Val_Metrics: ' + str(np.round(valid_metrics,4)) +
                    ')======'
                    )
                history_fold['best_epoch']              = e                    
                history_fold['best_train_losses']       = train_losses_mean
                history_fold['best_valid_losses']       = valid_losses_mean
                history_fold['best_valid_metrics']      = valid_metrics
                history_fold['best_valid_output']       = valid_output
                history_fold['best_valid_logits']       = valid_logits                    
                history_fold['best_valid_labels']       = valid_labels
                #score
                df_valid_output['eeg_ids']              = valid_eeg_ids
                df_valid_labels['eeg_ids']              = valid_eeg_ids
                df_valid_output['spectrogram_ids']      = valid_spectrogram_ids
                df_valid_labels['spectrogram_ids']      = valid_spectrogram_ids
                df_valid_output['patient_ids']          = valid_patient_ids
                df_valid_labels['patient_ids']          = valid_patient_ids
                col_labels_logits                       = [s.split('_')[0]+'_logits' for s in config['dataset']['col_labels']]
                df_valid_output[col_labels_logits]      = valid_logits
                history_fold['best_df_valid_output']    = df_valid_output
                history_fold['best_df_valid_labels']    = df_valid_labels

                #===other===#
                prob_pred                               = torch.tensor(valid_output)
                label                                   = torch.tensor(valid_labels)
                mae                                     = abs(prob_pred-label).mean(axis=0).tolist()
                acc                                     = []
                predicted_classes                       = torch.argmax(prob_pred, dim=1)
                label_classes                           = torch.argmax(label, dim=1)
                for c in range(prob_pred.shape[1]):
                    class_mask                          = label_classes == c
                    class_predictions                   = predicted_classes[class_mask]
                    class_labels                        = label_classes[class_mask]
                    correct_predictions                 = torch.eq(class_predictions, class_labels)
                    accuracy                            = torch.mean(correct_predictions.float())
                    acc.append(accuracy.item() * 100)

                history_fold['best_valid_MAE']          = mae
                history_fold['best_valid_ACC']          = acc
                formatted_mae = [f'{val:.2f}' for val in mae]
                formatted_acc = [f'{val:.1f}' for val in acc]
                LOGGER.info(f'======MAE : {formatted_mae}======')
                LOGGER.info(f'======ACC : {formatted_acc}======')

            if (e % 1000000== 0)&(e>1):
                metrics_str                             = "{:.4f}".format(valid_metrics).split('.')[1]
                torch.save(model, f'{save_output_dir}/best_EP{int(e)}_mAP{metrics_str}.pt')
                
    # Free memory
    del model
    history_fold['train_losses_epochs'] = train_losses_epochs
    history_fold['valid_losses_epochs'] = valid_losses_epochs
    history_fold['valid_metrics_epochs'] = valid_metrics_epochs
    return history_fold

################################################
def add_feat(df_feat,new_col_labels,setting_pp):
    add_ave                             = setting_pp['add_ave']
    add_max                             = setting_pp['add_max']
    add_min                             = setting_pp['add_min']
    add_std                             = setting_pp['add_std']
    add_count                           = setting_pp['add_count']
    add_delta                           = setting_pp['add_delta']
    new_col_labels_all                  = new_col_labels.copy()
    #patient_ave
    if add_ave:
        new_col_labels_patient_ave      = [s+'_patient_ave' for s in new_col_labels]
        patient_ave                     = df_feat.groupby('patient_id')[new_col_labels].mean().reset_index()
        patient_ave                     = patient_ave.rename(columns={s:s+'_patient_ave' for s in new_col_labels})
        df_feat                         = pd.merge(df_feat, patient_ave, on='patient_id')
        new_col_labels_all              += new_col_labels_patient_ave
    #patient_max
    if add_max:
        new_col_labels_patient_max      = [s+'_patient_max' for s in new_col_labels]
        patient_max                     = df_feat.groupby('patient_id')[new_col_labels].max().reset_index()
        patient_max                     = patient_max.rename(columns={s:s+'_patient_max' for s in new_col_labels})
        df_feat                         = pd.merge(df_feat, patient_max, on='patient_id')
        new_col_labels_all              += new_col_labels_patient_max
    #patient_min
    if add_min:
        new_col_labels_patient_min      = [s+'_patient_min' for s in new_col_labels]
        patient_min                     = df_feat.groupby('patient_id')[new_col_labels].min().reset_index()
        patient_min                     = patient_min.rename(columns={s:s+'_patient_min' for s in new_col_labels})
        df_feat                         = pd.merge(df_feat, patient_min, on='patient_id')
        new_col_labels_all              += new_col_labels_patient_min
    #patient_std
    if add_std:
        new_col_labels_patient_std      = [s+'_patient_std' for s in new_col_labels]
        patient_std                     = df_feat.groupby('patient_id')[new_col_labels].std().reset_index()
        patient_std                     = patient_std.rename(columns={s:s+'_patient_std' for s in new_col_labels})
        df_feat                         = pd.merge(df_feat, patient_std, on='patient_id')
        new_col_labels_all              += new_col_labels_patient_std
    #count_mesure
    if add_count:
        new_col_labels_patient_count    = ['count_mesure']
        patient_count                   = df_feat.groupby('patient_id').count().reset_index()
        patient_count                   = patient_count[['patient_id','eeg_id']].rename(columns={'eeg_id':'count_mesure'})
        df_feat                         = pd.merge(df_feat, patient_count, on='patient_id')
        new_col_labels_all              += new_col_labels_patient_count
    #patient_delta
    if add_delta:
        new_col_labels_patient_delta        = [s+'_patient_delta' for s in new_col_labels]
        df_feat[new_col_labels_patient_delta] = df_feat[new_col_labels].values - df_feat[new_col_labels_patient_ave].values
        new_col_labels_all                  += new_col_labels_patient_delta
    return df_feat,new_col_labels_all

#===base_feat===#
def preprocess_pp(dict_df_oof,setting_pp,col_labels_feat,
                  master_label,col_labels):
    use_ens                                 = setting_pp['use_ens']
    use_spec                                = setting_pp['use_spec']
    use_eeg_wave                            = setting_pp['use_eeg_wave']
    use_eeg_img                             = setting_pp['use_eeg_img']
    use_multi                               = setting_pp['use_multi']
    use_stacking                            = setting_pp['use_stacking']
    base_cols                               = ['eeg_id', 'spectrogram_id','patient_id', 'fold']
    try:
        df_feat                             = dict_df_oof['ens']
    except:
        try:
            df_feat                         = dict_df_oof['spec']
        except:
            df_feat                         = dict_df_oof['wave']
    df_feat                                 = df_feat[base_cols]
    uni_eeg_id                              = list(df_feat['eeg_id'].unique())
    uni_eeg_id.sort()
    new_col_labels                          = []
    if use_ens:
        df_oof_output_ens                   = dict_df_oof['ens']   
        new_col_labels_ens                  = [s+'_ens' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_ens
        df_feat.loc[:,new_col_labels_ens]   = df_oof_output_ens.loc[:,col_labels_feat].values
    if use_spec:
        df_oof_output_spec                  = dict_df_oof['spec']   
        new_col_labels_spec                 = [s+'_spec' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_spec
        df_feat.loc[:,new_col_labels_spec]  = df_oof_output_spec.loc[:,col_labels_feat].values
    if use_eeg_wave:
        df_oof_output_eeg_wave              = dict_df_oof['wave']  
        new_col_labels_eeg_wave             = [s+'_eeg_wave' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_eeg_wave
        df_feat.loc[:,new_col_labels_eeg_wave]  = df_oof_output_eeg_wave.loc[:,col_labels_feat].values
    if use_eeg_img:
        df_oof_output_eeg_img               = dict_df_oof['img']
        new_col_labels_eeg_img              = [s+'_eeg_img' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_eeg_img
        df_feat.loc[:,new_col_labels_eeg_img]   = df_oof_output_eeg_img.loc[:,col_labels_feat].values
    if use_multi:
        df_oof_output_multi                 = dict_df_oof['multi']
        new_col_labels_multi                = [s+'_multi' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_multi
        df_feat.loc[:,new_col_labels_multi] = df_oof_output_multi.loc[:,col_labels_feat].values
    if use_stacking:
        df_oof_output_stacking              = dict_df_oof['stack']
        new_col_labels_stacking             = [s+'_stacking' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_stacking
        df_feat.loc[:,new_col_labels_stacking] = df_oof_output_stacking.loc[:,col_labels_feat].values
        
    #===add_feat===#
    df_feat,new_col_labels_all              = add_feat(df_feat,new_col_labels,setting_pp)

    #===add_label===#
    df_feat[col_labels]                     = master_label[col_labels]
    df_feat                                 = df_feat.fillna(0)
    return df_feat,new_col_labels_all


def preprocess_pp_inference(dict_df_oof,setting_pp,col_labels_feat):
    use_ens                                 = setting_pp['use_ens']
    use_spec                                = setting_pp['use_spec']
    use_eeg_wave                            = setting_pp['use_eeg_wave']
    use_eeg_img                             = setting_pp['use_eeg_img']
    use_multi                               = setting_pp['use_multi']
    use_stacking                            = setting_pp['use_stacking']
    base_cols                               = ['eeg_id', 'spectrogram_id','patient_id']
    df_feat                                 = dict_df_oof['ens']
    df_feat                                 = df_feat[base_cols]
    new_col_labels                          = []
    if use_ens:
        df_oof_output_ens                   = dict_df_oof['ens']   
        new_col_labels_ens                  = [s+'_ens' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_ens
        df_feat.loc[:,new_col_labels_ens]   = df_oof_output_ens.loc[:,col_labels_feat].values
    if use_spec:
        df_oof_output_spec                  = dict_df_oof['spec']   
        new_col_labels_spec                 = [s+'_spec' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_spec
        df_feat.loc[:,new_col_labels_spec]  = df_oof_output_spec.loc[:,col_labels_feat].values
    if use_eeg_wave:
        df_oof_output_eeg_wave              = dict_df_oof['wave']  
        new_col_labels_eeg_wave             = [s+'_eeg_wave' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_eeg_wave
        df_feat.loc[:,new_col_labels_eeg_wave]  = df_oof_output_eeg_wave.loc[:,col_labels_feat].values
    if use_eeg_img:
        df_oof_output_eeg_img               = dict_df_oof['img']
        new_col_labels_eeg_img              = [s+'_eeg_img' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_eeg_img
        df_feat.loc[:,new_col_labels_eeg_img]   = df_oof_output_eeg_img.loc[:,col_labels_feat].values
    if use_multi:
        df_oof_output_multi                 = dict_df_oof['multi']
        new_col_labels_multi                = [s+'_multi' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_multi
        df_feat.loc[:,new_col_labels_multi] = df_oof_output_multi.loc[:,col_labels_feat].values
    if use_stacking:
        df_oof_output_stacking              = dict_df_oof['stack']
        new_col_labels_stacking             = [s+'_stacking' for s in col_labels_feat]
        new_col_labels                      += new_col_labels_stacking
        df_feat.loc[:,new_col_labels_stacking] = df_oof_output_stacking.loc[:,col_labels_feat].values
    #===add_feat===#
    df_feat,new_col_labels_all              = add_feat(df_feat,new_col_labels,setting_pp)

    df_feat                                 = df_feat.fillna(0)
    return df_feat,new_col_labels_all
