# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# ===============
# import mycode
# ===============
import sys
sys.path.append('../')
from trainer.metrics  import score

# ===========================================================================
# Model_Train
# ===========================================================================
def calc_train(model, config, train_dataset, val_dataset,train_loader, val_loader,
                optimizer,scheduler,loss_function,save_output_dir,LOGGER,
                history_fold,
                df_fold_valid):
    #=params=#   
    input_type           = config['dataset']['input_type']
    model_type           = config['model']['model_type']
    mode_usepretrain     = config['calc_mode']['mode_usepretrain']
    DEVICE               = config['train']['DEVICE']
    n_epochs             = config['train']['EPOCHS']
    select_data_epoch           = config['dataset']['select_data_epoch']
    
    #eeg_wave
    calc_select_time2_wave      = config['dataset']['EEG']['select_time2_wave']['calc_select_time2_wave']
    use_time_ave_wave           = config['dataset']['EEG']['select_time2_wave']['use_time_ave_wave']
    weights_time_wave           = config['dataset']['EEG']['select_time2_wave']['weights_time_wave']
    #eeg_img
    calc_select_time2_img       = config['dataset']['EEG_IMG']['select_time2_img']['calc_select_time2_img']
    use_time_ave_img            = config['dataset']['EEG_IMG']['select_time2_img']['use_time_ave_img']
    weights_time_img            = config['dataset']['EEG_IMG']['select_time2_img']['weights_time_img']

    #select_data
    select_train_data           = config['calc_mode']['select_train_data']
    select_valid_data           = config['calc_mode']['select_valid_data']
    calc_valid_result           = config['calc_mode']['calc_valid_result']
    judge_valid_result          = config['calc_mode']['judge_valid_result']
    dict_labelid2flg1vote       = df_fold_valid.set_index('label_id')['flg1_vote'].to_dict()

    #mixup
    mixup_prob_spec             = config['dataset']['SPEC']['mixup']['prob']
    mixup_alpha_spec            = config['dataset']['SPEC']['mixup']['alpha']
    mixup_prob_eeg_img          = config['dataset']['EEG_IMG']['mixup']['prob']
    mixup_alpha_eeg_img         = config['dataset']['EEG_IMG']['mixup']['alpha']
    mixup_prob_eeg_wave         = config['dataset']['EEG']['mixup']['prob']
    mixup_alpha_eeg_wave        = config['dataset']['EEG']['mixup']['alpha']
    stop_mixup_epoch_spec       = config['dataset']['SPEC']['mixup']['stop_mixup_epoch']
    stop_mixup_epoch_eeg_img    = config['dataset']['EEG_IMG']['mixup']['stop_mixup_epoch']
    stop_mixup_epoch_eeg_wave   = config['dataset']['EEG']['mixup']['stop_mixup_epoch']
    mixup_prob_ALL              = config['dataset']['ALL']['mixup']['prob']
    mixup_alpha_ALL             = config['dataset']['ALL']['mixup']['alpha']
    stop_mixup_epoch_ALL        = config['dataset']['ALL']['mixup']['stop_mixup_epoch']

    #time_cutmix
    cutmix_prob_spec            = config['dataset']['SPEC']['time_cutmix']['prob']
    cutmix_alpha_spec           = config['dataset']['SPEC']['time_cutmix']['alpha']
    cutmix_prob_eeg_img         = config['dataset']['EEG_IMG']['time_cutmix']['prob']
    cutmix_alpha_eeg_img        = config['dataset']['EEG_IMG']['time_cutmix']['alpha']
    cutmix_prob_eeg_wave        = config['dataset']['EEG']['time_cutmix']['prob']
    cutmix_alpha_eeg_wave       = config['dataset']['EEG']['time_cutmix']['alpha']
    cutmix_mode_start0_eeg_wave = config['dataset']['EEG']['time_cutmix']['mode_start0']
    stop_cutmix_epoch_spec      = config['dataset']['SPEC']['time_cutmix']['stop_cutmix_epoch']
    stop_cutmix_epoch_eeg_img   = config['dataset']['EEG_IMG']['time_cutmix']['stop_cutmix_epoch']
    stop_cutmix_epoch_eeg_wave  = config['dataset']['EEG']['time_cutmix']['stop_cutmix_epoch']
    cutmix_prob_ALL             = config['dataset']['ALL']['time_cutmix']['prob']
    cutmix_alpha_ALL            = config['dataset']['ALL']['time_cutmix']['alpha']
    stop_cutmix_epoch_ALL       = config['dataset']['ALL']['time_cutmix']['stop_cutmix_epoch']

    #multi_model
    MULTI_calc_spec             = config['model']['MULTI']['calc_spec']
    MULTI_calc_eeg_wave         = config['model']['MULTI']['calc_eeg_wave']
    MULTI_calc_eeg_img          = config['model']['MULTI']['calc_eeg_img']
    
    #=logger=#
    LOGGER.info(f"===input_type={input_type}===")
    LOGGER.info(f"===model_type={model_type}===")
    LOGGER.info(f"===mode_usepretrain={mode_usepretrain}===")
    if mode_usepretrain:
        exp_id_load         = config['model']['LOAD']['exp_id_load']
        LOGGER.info(f"===exp_id_load={exp_id_load}===")
    LOGGER.info(f"===select_train_data={select_train_data}===")
    LOGGER.info(f"===select_valid_data={select_valid_data}===")
    LOGGER.info(f"===calc_valid_result={calc_valid_result}===")
    LOGGER.info(f"===judge_valid_result={judge_valid_result}===")
    LOGGER.info('===Starting training loop===')
    LOGGER.info("EPOCHS={}".format(n_epochs))
    #=initialize1:metric=#
    best_corrects           = 99999
    # ===============
    # Train loop
    # ===============
    train_losses_epochs = []
    valid_losses_epochs = []
    valid_metrics_epochs= []
    for e in range(0, n_epochs):        
        if select_data_epoch:
            train_dataset.update_for_epoch(e)
        # ===============
        # Train
        # ===============
        train_losses                = []
        model.train()
        for batch, data in enumerate(train_loader):
            # LOGGER.info(batch)
            label_id                = data['label_id']
            patient_id              = data['patient_id']
            eeg_id                  = data['eeg_id']
            spectrogram_id          = data['spectrogram_id']
            label                   = data['label']
            #input
            if input_type=='EEG_WAVE':
                feature             = data['sub_eeg_data_wave']
                #time_cutmix
                dice1               = (np.random.random() < mixup_prob_eeg_wave)&(e<stop_mixup_epoch_eeg_wave)
                if dice1:
                    feature,label   = Mixup_nn(feature, label, mixup_alpha_eeg_wave)
                #time_cutmix
                dice2               = (np.random.random() < cutmix_prob_eeg_wave)&(e<stop_cutmix_epoch_eeg_wave)
                if dice2:
                    feature,label   = Time_CutMixup_nn(feature, label, cutmix_alpha_eeg_wave,
                                                       cutmix_mode_start0_eeg_wave)

                if torch.cuda.is_available():
                    feature         = feature.float().to(DEVICE)
                    label           = label.to(DEVICE)
            elif input_type=='EEG_IMG':
                feature             = data['sub_eeg_data_img']
                #mixup
                dice1               = (np.random.random() < mixup_prob_eeg_img)&(e<stop_mixup_epoch_eeg_img)
                if dice1:
                    feature,label   = Mixup_nn(feature, label, mixup_alpha_eeg_img)
                #time_cutmix
                dice2               = (np.random.random() < cutmix_prob_eeg_img)&(e<stop_cutmix_epoch_eeg_img)
                if dice2:
                    feature,label   = Time_CutMixup_nn(feature, label, cutmix_alpha_eeg_img)

                if torch.cuda.is_available():
                    feature         = feature.float().to(DEVICE)
                    label           = label.to(DEVICE)

            elif input_type=='SPEC':
                feature             = data['sub_spectrogram_4ch']
                #mixup
                dice1               = (np.random.random() < mixup_prob_spec)&(e<stop_mixup_epoch_spec)
                if dice1:
                    feature,label   = Mixup_nn(feature, label, mixup_alpha_spec)
                #time_cutmix
                dice2                = (np.random.random() < cutmix_prob_spec)&(e<stop_cutmix_epoch_spec)
                if dice2:
                    feature,label   = Time_CutMixup_nn(feature, label, cutmix_alpha_spec)

                if torch.cuda.is_available():
                    feature         = feature.float().to(DEVICE)
                    label           = label.to(DEVICE)
            elif input_type=='ALL':
                if MULTI_calc_spec:
                    feature_spec    = data['sub_spectrogram_4ch']
                else:
                    feature_spec    = []
                if MULTI_calc_eeg_img:
                    feature_eeg_img = data['sub_eeg_data_img']
                else:
                    feature_eeg_img = []
                if MULTI_calc_eeg_wave:
                    feature_eeg_wave= data['sub_eeg_data_wave']
                else:
                    feature_eeg_wave= []
                #time_cutmix
                dice1               = (np.random.random() < mixup_prob_ALL)&(e<stop_mixup_epoch_ALL)
                if dice1:
                    feature_spec, feature_eeg_img,feature_eeg_wave,label    = Mixup_nn_multi(feature_spec, feature_eeg_img,feature_eeg_wave,
                                                                                            label, mixup_alpha_ALL,
                                                                                            MULTI_calc_spec,MULTI_calc_eeg_wave,MULTI_calc_eeg_img)
                #time_cutmix
                dice2               = (np.random.random() < cutmix_prob_ALL)&(e<stop_cutmix_epoch_ALL)
                if dice2:
                    feature_spec, feature_eeg_img,feature_eeg_wave,label    = Time_CutMixup_nn_multi(feature_spec, feature_eeg_img,feature_eeg_wave,
                                                                                                    label, cutmix_alpha_ALL,
                                                                                                    MULTI_calc_spec,MULTI_calc_eeg_wave,MULTI_calc_eeg_img)
                if torch.cuda.is_available():
                    if MULTI_calc_spec:
                        feature_spec    = feature_spec.float().to(DEVICE)
                    if MULTI_calc_eeg_img:
                        feature_eeg_img = feature_eeg_img.float().to(DEVICE)
                    if MULTI_calc_eeg_wave:
                        feature_eeg_wave= feature_eeg_wave.float().to(DEVICE)
                    label           = label.to(DEVICE)

            #=gradient_zero_clear=
            optimizer.zero_grad() 
            #=preds=
            if model_type == 'CNN':
                (_,logits)              = model(feature)
            elif (model_type == 'WAVE_CNN'):
                (_,logits)              = model(feature,e)
            elif model_type == 'MULTI':
                (_,logits)          = model(feature_spec,
                                                feature_eeg_wave,
                                                feature_eeg_img)
            pred                        = nn.functional.log_softmax(logits, dim=1)
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
            train_losses_mean               = np.mean(np.array(train_losses))  
            LOGGER.info('Ep ' + str(e) + ', ==Train END==, Train_Loss: ' + str(np.round(train_losses_mean,4)))
        # ===============
        # Valid
        # =============== 
        with torch.no_grad():
            #=init=        
            model.eval()
            #=batch_loop=#
            for batch, data in enumerate(val_loader):
                label_id                = data['label_id']
                patient_id              = data['patient_id']
                eeg_id                  = data['eeg_id']
                spectrogram_id          = data['spectrogram_id']
                label                   = data['label']#label [4, 5760, 3]
                #input
                if input_type=='EEG_WAVE':
                    feature             = data['sub_eeg_data_wave']
                    if torch.cuda.is_available():
                        label           = label.to(DEVICE)
                elif input_type=='EEG_IMG':
                    feature             = data['sub_eeg_data_img']
                    if torch.cuda.is_available():
                        label           = label.to(DEVICE)
                elif input_type=='SPEC':
                    feature             = data['sub_spectrogram_4ch']
                    if torch.cuda.is_available():
                        label           = label.to(DEVICE)
                elif input_type=='ALL':
                    if MULTI_calc_spec:
                        feature_spec    = data['sub_spectrogram_4ch']
                    else:
                        feature_spec    = []
                    if MULTI_calc_eeg_img:
                        feature_eeg_img = data['sub_eeg_data_img']
                    else:
                        feature_eeg_img = []
                    if MULTI_calc_eeg_wave:
                        feature_eeg_wave= data['sub_eeg_data_wave']
                    else:
                        feature_eeg_wave= []
                    if torch.cuda.is_available():
                        label               = label.to(DEVICE)
                        
                #=preds=
                if model_type == 'CNN':
                    if (calc_select_time2_img==True)&(use_time_ave_img==True):
                        length              = feature.shape[-1]//5
                        for idt in range(5):
                            feature_tmp     = feature[:,:,:,:,idt*length:(idt+1)*length].float().to(DEVICE)
                            (_,logits_tmp)  = model(feature_tmp)
                            if idt ==0:
                                logits      = logits_tmp*weights_time_img[idt]
                            else:
                                logits      += logits_tmp*weights_time_img[idt]
                    else:
                        (_,logits)          = model(feature.float().to(DEVICE))

                elif (model_type == 'WAVE_CNN') :
                    if (calc_select_time2_wave==True)&(use_time_ave_wave==True):
                        length              = 2000
                        for idt in range(5):
                            feature_tmp     = feature[:,:,idt*length:(idt+1)*length].float().to(DEVICE)
                            (_,logits_tmp)  = model(feature_tmp,e)
                            if idt ==0:
                                logits      = logits_tmp*weights_time_wave[idt]
                            else:
                                logits      += logits_tmp*weights_time_wave[idt]
                    else:
                        (_,logits)          = model(feature.float().to(DEVICE),e)

                elif model_type == 'MULTI':
                    if (calc_select_time2_wave==True)&(use_time_ave_wave==True)&\
                        (calc_select_time2_img==True)&(use_time_ave_img==True):#waveもimgも分割する

                        length_wave         = 2000
                        length_img          = feature_eeg_img.shape[-1]//5

                        if MULTI_calc_spec:
                            feature_spec    = feature_spec.float().to(DEVICE)

                        for idt in range(5):
                            if MULTI_calc_eeg_img:
                                feature_eeg_img_tmp     = feature_eeg_img[:,:,:,:,idt*length_img:(idt+1)*length_img].float().to(DEVICE)
                            if MULTI_calc_eeg_wave:
                                feature_eeg_wave_tmp    = feature_eeg_wave[:,:,idt*length_wave:(idt+1)*length_wave].float().to(DEVICE)
                                
                            (_,logits_tmp)  = model(feature_spec,
                                                        feature_eeg_wave_tmp,
                                                        feature_eeg_img_tmp) 
                            if idt ==0:
                                logits      = logits_tmp*weights_time_wave[idt]
                            else:
                                logits      += logits_tmp*weights_time_wave[idt]

                    elif (calc_select_time2_wave==True)&(use_time_ave_wave==True)&\
                        (calc_select_time2_img==False)&(use_time_ave_img==False):#waveは分割するけど、imgはしない
                        length_wave         = 2000
                        if MULTI_calc_spec:
                            feature_spec    = feature_spec.float().to(DEVICE)
                        if MULTI_calc_eeg_img:
                            feature_eeg_img = feature_eeg_img.float().to(DEVICE)
                        for idt in range(5):
                            if MULTI_calc_eeg_wave:
                                feature_eeg_wave_tmp    = feature_eeg_wave[:,:,idt*length_wave:(idt+1)*length_wave].float().to(DEVICE)                                        
                            (_,logits_tmp)  = model(feature_spec,
                                                        feature_eeg_wave_tmp,
                                                        feature_eeg_img) 
                            if idt ==0:
                                logits      = logits_tmp*weights_time_wave[idt]
                            else:
                                logits      += logits_tmp*weights_time_wave[idt]

                    elif (calc_select_time2_wave==False)&(use_time_ave_wave==False)&\
                        (calc_select_time2_img==True)&(use_time_ave_img==True):#waveもimgも分割する
                        length_img          = feature_eeg_img.shape[-1]//5
                        if MULTI_calc_spec:
                            feature_spec    = feature_spec.float().to(DEVICE)
                        if MULTI_calc_eeg_wave:
                            feature_eeg_wave= feature_eeg_wave.float().to(DEVICE)
                        for idt in range(5):
                            if MULTI_calc_eeg_img:
                                feature_eeg_img_tmp     = feature_eeg_img[:,:,:,:,idt*length_img:(idt+1)*length_img].float().to(DEVICE)
                                
                            (_,logits_tmp)  = model(feature_spec,
                                                    feature_eeg_wave,
                                                    feature_eeg_img_tmp) 
                            if idt ==0:
                                logits      = logits_tmp*weights_time_img[idt]
                            else:
                                logits      += logits_tmp*weights_time_img[idt]

                    else:
                        if MULTI_calc_spec:
                            feature_spec    = feature_spec.float().to(DEVICE)
                        if MULTI_calc_eeg_img:
                            feature_eeg_img = feature_eeg_img.float().to(DEVICE)
                        if MULTI_calc_eeg_wave:
                            feature_eeg_wave= feature_eeg_wave.float().to(DEVICE)
                        (_,logits)      = model(feature_spec,
                                                    feature_eeg_wave,
                                                    feature_eeg_img)  
                            
                pred                        = nn.functional.log_softmax(logits, dim=1)                    
                #=loss=
                loss                        = loss_function(pred, label).mean()
                if batch==0:
                    valid_label_ids         = label_id.tolist()
                    valid_patient_ids       = patient_id.tolist()
                    valid_eeg_ids           = eeg_id.tolist()
                    valid_spectrogram_ids   = spectrogram_id.tolist()
                    valid_logits            = logits.float().detach().cpu()
                    valid_output            = nn.functional.softmax(logits, dim=1).float().detach().cpu()
                    valid_labels            = label.float().detach().cpu()
                    valid_losses            = [loss.float().detach().cpu()]
                else:
                    valid_label_ids         += label_id.tolist()
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
            valid_logits                    = valid_logits.numpy()
            valid_output                    = valid_output.numpy()
            valid_labels                    = valid_labels.numpy()
            #=score=#
            df_valid_output                 = pd.DataFrame(valid_output,columns=config['dataset']['col_labels'])
            df_valid_labels                 = pd.DataFrame(valid_labels,columns=config['dataset']['col_labels'])

            str_result                              =  'Ep ' + str(e) + ', ==Val END==, Val_Loss: ' + str(np.round(valid_losses_mean,4))  
            #score_All
            if 'All' in calc_valid_result:
                df_valid_output_all                 = df_valid_output.copy()
                df_valid_labels_all                 = df_valid_labels.copy()
                df_valid_output_all['id']           = np.arange(len(df_valid_output_all))
                df_valid_labels_all['id']           = np.arange(len(df_valid_labels_all))
                valid_metrics_all                   = score(solution=df_valid_labels_all.copy(), submission=df_valid_output_all.copy(), 
                                                            row_id_column_name='id')
                
                str_result                          += ', Val_Metrics_ALL: ' + str(np.round(valid_metrics_all,4)) +', len_ALL: ' + str(len(df_valid_output_all)) 

            #score_flg1
            if 'flg1_vote' in calc_valid_result:
                df_valid_output_flg1                = df_valid_output.copy()
                df_valid_labels_flg1                = df_valid_labels.copy()
                df_valid_output_flg1['label_ids']   = valid_label_ids
                df_valid_labels_flg1['label_ids']   = valid_label_ids
                df_valid_output_flg1['flg1_vote']   = df_valid_output_flg1['label_ids'].map(dict_labelid2flg1vote)
                select_rule_flg1                    = df_valid_output_flg1['flg1_vote'].values
                df_valid_output_flg1                = df_valid_output_flg1[select_rule_flg1].reset_index(drop=True)
                df_valid_labels_flg1                = df_valid_labels_flg1[select_rule_flg1].reset_index(drop=True)
                df_valid_output_flg1                = df_valid_output_flg1.drop(columns=['label_ids','flg1_vote'])
                df_valid_labels_flg1                = df_valid_labels_flg1.drop(columns=['label_ids'])
                df_valid_output_flg1['id']          = np.arange(len(df_valid_output_flg1))
                df_valid_labels_flg1['id']          = np.arange(len(df_valid_labels_flg1))
                valid_metrics_flg1_vote             = score(solution=df_valid_labels_flg1.copy(), submission=df_valid_output_flg1.copy(), 
                                                            row_id_column_name='id')
                str_result                          += ', Val_Metrics_flg1_vote: ' + str(np.round(valid_metrics_flg1_vote,4))+', len_flg1_vote: ' + str(len(df_valid_output_flg1))
                
            LOGGER.info(str_result)  
                
            #score_judge
            if judge_valid_result=='All':
                valid_metrics                       = valid_metrics_all
            elif judge_valid_result=='flg1_vote':
                valid_metrics                       = valid_metrics_flg1_vote

            train_losses_epochs.append(train_losses_mean)
            valid_losses_epochs.append(valid_losses_mean)
            valid_metrics_epochs.append(valid_metrics)
            # =============================
            # Save model_best  Valid
            # =============== ==============
            if valid_metrics <= best_corrects:
                best_corrects                       = valid_metrics
                try:
                    torch.save(model, save_output_dir+'/best.pt')
                except:#custom_classの場合
                    torch.save(model.state_dict(), save_output_dir+'/best_state_dict.pt')
                LOGGER.info('======Saving new best model at epoch ' + str(e) +
                    ',  Val_Loss: ' + str(np.round(valid_losses_mean,4))+', Val_Metrics: ' + str(np.round(valid_metrics,4)) +
                    ')======'
                    )
                history_fold['best_epoch']              = e                    
                history_fold['best_train_losses']       = train_losses_mean
                history_fold['best_valid_losses']       = valid_losses_mean
                history_fold['best_valid_metrics']      = valid_metrics
                history_fold['best_valid_label_ids']    = valid_label_ids
                history_fold['best_valid_output']       = valid_output
                history_fold['best_valid_logits']       = valid_logits                    
                history_fold['best_valid_labels']       = valid_labels
                #score
                df_valid_output['label_ids']            = valid_label_ids
                df_valid_labels['label_ids']            = valid_label_ids
                df_valid_output['eeg_ids']              = valid_eeg_ids
                df_valid_labels['eeg_ids']              = valid_eeg_ids
                df_valid_output['spectrogram_ids']      = valid_spectrogram_ids
                df_valid_labels['spectrogram_ids']      = valid_spectrogram_ids
                df_valid_output['patient_ids']          = valid_patient_ids
                df_valid_labels['patient_ids']          = valid_patient_ids
                #flg
                df_valid_output['flg1_vote']            = df_valid_output['label_ids'].map(dict_labelid2flg1vote)
                df_valid_labels['flg1_vote']            = df_valid_labels['label_ids'].map(dict_labelid2flg1vote)

                #選定
                df_valid_output_all                     = df_valid_output.copy()
                df_valid_labels_all                     = df_valid_labels.copy()
                if judge_valid_result=='All':
                    df_valid_output                     = df_valid_output.copy()
                    df_valid_labels                     = df_valid_labels.copy()
                elif judge_valid_result=='flg1_vote':
                    df_valid_output                     = df_valid_output[df_valid_output['flg1_vote']].reset_index(drop=True)
                    df_valid_labels                     = df_valid_labels[df_valid_labels['flg1_vote']].reset_index(drop=True)

                history_fold['best_df_valid_output']    = df_valid_output
                history_fold['best_df_valid_labels']    = df_valid_labels
                history_fold['best_df_valid_output_all']= df_valid_output_all
                history_fold['best_df_valid_labels_all']= df_valid_labels_all
                
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
    # Free memory
    del model
    #history 
    history_fold['train_losses_epochs'] = train_losses_epochs
    history_fold['valid_losses_epochs'] = valid_losses_epochs
    history_fold['valid_metrics_epochs'] = valid_metrics_epochs
    return history_fold


#############################################################
def get_fig(history_fold,save_output_dir_fold,config):
    n_epochs             = config['train']['EPOCHS']
    fig=plt.figure(figsize=(10,10))
    #1.loss
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(np.arange(n_epochs)+1,history_fold['train_losses_epochs'],'r')
    ax1.plot(np.arange(n_epochs)+1,history_fold['valid_losses_epochs'],'b')
    ax1.set_title('Loss:best_val'+str(np.round(history_fold['best_valid_losses'],3)))
    ax1.set_ylabel('loss')
    ax1.set_xlabel('Epoch')
    ax1.grid('both')
    ax1.legend(['Train_loss','Val_loss'])
    #2.metrics
    ax2=fig.add_subplot(2,1,2)
    ax2.plot(np.arange(n_epochs)+1,history_fold['valid_metrics_epochs'],'b')
    ax2.set_title('metrics:best_val'+str(np.round(history_fold['best_valid_metrics'],3)))
    ax2.set_ylabel('metrics')
    ax2.set_xlabel('Epoch')
    ax2.grid('both')
    ax2.legend(['Val_metrics'])
    plt.subplots_adjust(hspace=0.4) 
    plt.savefig(f'{save_output_dir_fold}/history')

# ==============================
# Mix_Up
# ==============================
from torch.distributions import Beta
def do_mixup(X,perm,coeffs):
    n_dims              = len(X.shape)
    if n_dims == 2:
        X               = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
    elif n_dims == 3:
        X               = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
    elif n_dims == 4:
        X               = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]#もとのとランダム並び替えをmixup
    else:
        X               = coeffs.view(-1, 1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1, 1)) * X[perm]#もとのとランダム並び替えをmixup
    return X

def Mixup_nn(X, Y, mix_beta):
    beta_distribution   = Beta(mix_beta, mix_beta)
    bs                  = X.shape[0]#6
    perm                = torch.randperm(bs)#[0,1,2,3,4,5]をrandom並び替え
    coeffs              = beta_distribution.rsample(torch.Size((bs,))).to(X.device)
    X                   = do_mixup(X,perm,coeffs)
    Y                   = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
    return X, Y

def Mixup_nn_multi(X1,X2,X3, Y, mix_beta,
                   MULTI_calc_spec,MULTI_calc_eeg_wave,MULTI_calc_eeg_img):#X1:spec,X2:img,X3:wave
    beta_distribution   = Beta(mix_beta, mix_beta)
    if MULTI_calc_spec:
        bs              = X1.shape[0]#6
        device          = X1.device
    elif MULTI_calc_eeg_wave:
        bs              = X3.shape[0]#6
        device          = X3.device
    elif MULTI_calc_eeg_img:
        bs              = X2.shape[0]#6
        device          = X2.device 
    perm                = torch.randperm(bs)#[0,1,2,3,4,5]をrandom並び替え
    coeffs              = beta_distribution.rsample(torch.Size((bs,))).to(device)
    
    if MULTI_calc_spec:
        X1              = do_mixup(X1,perm,coeffs)
    if MULTI_calc_eeg_wave:
        pass
        # X3                  = do_mixup(X3,perm,coeffs) waveはmixupしない
    if MULTI_calc_eeg_img:
        X2              = do_mixup(X2,perm,coeffs)
    Y                   =  (2*coeffs.view(-1, 1)+1)/3 * Y + 2*(1 - coeffs.view(-1, 1))/3 * Y[perm]
    return X1,X2,X3, Y

# ==============================
def do_cutmix(X,perm,coeffs,mode_start0):
    n_dims              = len(X.shape)
    tmp                 = torch.zeros_like(X)
    if n_dims == 2:
        time_length             = X.shape[1]
        cut_time_length         = int(time_length * coeffs)
        if mode_start0==False:
            start_pos           = torch.randint(0, time_length - cut_time_length, (1,)).item()
        else:
            start_pos           = 0
        tmp[:, :start_pos]                              = X[perm, :start_pos]
        tmp[:, start_pos:start_pos+cut_time_length]     = X[:, start_pos:start_pos+cut_time_length]
        tmp[:, start_pos+cut_time_length:]              = X[perm, start_pos+cut_time_length:]
    elif n_dims == 3:
        time_length             = X.shape[2]
        cut_time_length         = int(time_length * coeffs)
        if mode_start0==False:
            start_pos           = torch.randint(0, time_length - cut_time_length, (1,)).item()
        else:
            start_pos           = 0
        tmp[:,:, :start_pos]                            = X[perm,:, :start_pos]
        tmp[:,:, start_pos:start_pos+cut_time_length]   = X[:,:, start_pos:start_pos+cut_time_length]
        tmp[:,:, start_pos+cut_time_length:]            = X[perm,:, start_pos+cut_time_length:]
    elif n_dims == 4:
        time_length             = X.shape[3]
        cut_time_length         = int(time_length * coeffs)
        if mode_start0==False:
            start_pos           = torch.randint(0, time_length - cut_time_length, (1,)).item()
        else:
            start_pos           = 0
        tmp[:,:,:, :start_pos]                          = X[perm,:,:, :start_pos]
        tmp[:,:,:, start_pos:start_pos+cut_time_length] = X[:,:,:, start_pos:start_pos+cut_time_length]
        tmp[:,:,:, start_pos+cut_time_length:]          = X[perm,:,:, start_pos+cut_time_length:]
    else:
        time_length             = X.shape[3]
        cut_time_length         = int(time_length * coeffs)
        if mode_start0==False:
            start_pos           = torch.randint(0, time_length - cut_time_length, (1,)).item()
        else:
            start_pos           = 0
        tmp[:,:,:,:, :start_pos]                          = X[perm,:,:,:, :start_pos]
        tmp[:,:,:,:, start_pos:start_pos+cut_time_length] = X[:,:,:,:, start_pos:start_pos+cut_time_length]
        tmp[:,:,:,:, start_pos+cut_time_length:]          = X[perm,:,:,:, start_pos+cut_time_length:]
    X                   = tmp
    return X

def Time_CutMixup_nn(X, Y, mix_beta,mode_start0=False):
    beta_distribution   = Beta(mix_beta, mix_beta)
    bs                  = X.shape[0]#6
    perm                = torch.randperm(bs)#
    coeffs              = beta_distribution.rsample(torch.Size((1,)))
    X                   = do_cutmix(X,perm,coeffs,mode_start0)
    Y                   = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
    return X, Y

def Time_CutMixup_nn_multi(X1,X2,X3, Y, mix_beta,
                            MULTI_calc_spec,MULTI_calc_eeg_wave,MULTI_calc_eeg_img,
                            mode_start0=False):
    beta_distribution   = Beta(mix_beta, mix_beta)
    if MULTI_calc_spec:
        bs              = X1.shape[0]#6
    elif MULTI_calc_eeg_wave:
        bs              = X3.shape[0]#6
    elif MULTI_calc_eeg_img:
        bs              = X2.shape[0]#6
    perm                = torch.randperm(bs)#
    coeffs              = beta_distribution.rsample(torch.Size((1,)))
    if MULTI_calc_spec:
        X1              = do_cutmix(X1,perm,coeffs,mode_start0)
    if MULTI_calc_eeg_wave:
        X3              = do_cutmix(X3,perm,coeffs,mode_start0)
    if MULTI_calc_eeg_img:
        X2              = do_cutmix(X2,perm,coeffs,mode_start0)
    Y                   = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
    return X1,X2,X3, Y