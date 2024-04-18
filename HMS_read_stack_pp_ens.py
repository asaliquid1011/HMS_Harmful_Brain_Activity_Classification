#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    import os
    new_directory = "workspace/"
    os.chdir(new_directory)
except:
    pass


# In[2]:


# =============================
# Import
# =============================
#system    
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import yaml
import importlib
#the basics
import pandas as pd
import torch

# ===============
# import mycode
# ===============
sys.path.append("./src")
#utils
from utils.utils import seed_everything,AttrDict
from utils.logger import setup_logger, LOGGER
# ===============
# Read yaml     
# ===============
from argparse import ArgumentParser
try:
    parser       = ArgumentParser()
    parser.add_argument('--config', type=str)
    args         = parser.parse_args()
    config_name  = args.config
except:
    config_name = None
    
if config_name==None:
    with open('./yaml/config_HMS_read_stack_pp_ens.yaml', 'r') as yml:
        config              = yaml.safe_load(yml)
else:
    with open('./yaml/'+config_name+'.yaml', 'r') as yml:
        config             = yaml.safe_load(yml)
# ===============
# Path_Base     
# ===============
EXP_ID                  = config['EXP_ID']
LOGGER_dir              = config['path']['LOGGER_dir']
LOGGER_PATH             = LOGGER_dir+'/log_'+EXP_ID+".txt"
input_dir               = config['path']['input_dir']
data_dir                = config['path']['data_dir']
output_dir              = config['path']['output_dir']
save_output_dir         = f'{output_dir}/{EXP_ID}'
os.makedirs(save_output_dir,exist_ok=True)
# ===============
# utils
# ===============
seed                    = config['train']['SEED']
seed_everything(seed)
# ===============
# LOGGER
# ===============
setup_logger(out_file=LOGGER_PATH)
# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#AttrDict
config             = AttrDict(config)


# # Setting 

# In[3]:


#setting
col_labels              = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
col_labels_logits       = [s.split('_')[0]+'_logits' for s in col_labels]
select_id               = config['calc_mode']['select_id']
select_col_label        = config['calc_mode']['select_col_label']
data_type               = config['calc_mode']['data_type']
if select_col_label =='proba':
    col_labels_feat     = col_labels
elif select_col_label =='logits':
    col_labels_feat     = col_labels_logits
fold_type               = config['calc_mode']['fold_type']
if fold_type == 'TH':
    if data_type =='ALL':
        dir_master      = 'master_FOLD_TH_exp962_ALL'
        all_mode        = True
    else:
        dir_master      = 'master_FOLD_TH_exp963'
        all_mode        = False
    exp_id_fold         = 'exp_id_fold_TH'
elif fold_type == 'THV2':
    if data_type =='ALL':
        dir_master      = 'master_FOLD_TH_exp970_ALL'
        all_mode        = True
    else:
        dir_master      = 'master_FOLD_THV2_exp971'
        all_mode        = False
    exp_id_fold         = 'exp_id_fold_THV2'
exp_id_spec             = config[exp_id_fold]['spec']
exp_id_eeg_wave         = config[exp_id_fold]['eeg_wave']
exp_id_eeg_img          = config[exp_id_fold]['eeg_img']
exp_id_multi            = config[exp_id_fold]['multi']
config['calc_mode']['Calc_Fold'] = config[exp_id_fold]['Calc_Fold']

stacking_mode           = config['calc_mode']['stacking_mode']
calc_stack              = config['calc_mode']['calc_stack']
calc_pp                 = config['calc_mode']['calc_pp']

label_mode              = config['calc_mode']['label_mode']


# # 1_Read_Data_Save  

# In[4]:


import postprocess.utils_read_stack_pp_ens
importlib.reload(postprocess.utils_read_stack_pp_ens)
from postprocess.utils_read_stack_pp_ens import read_result,softmax_second_axis

#read_master
if label_mode == 'specified':
    master_label            = pd.read_csv(f'{output_dir}/{dir_master}/df_oof_labels.csv')
    master_meta             = pd.read_csv(f'{output_dir}/{dir_master}/train_meta.csv')
    master_label            = master_label.drop(columns=['Unnamed: 0','flg2_kl'])
    master_label            = master_label.rename(columns={'label_ids':'label_id','eeg_ids':'eeg_id','spectrogram_ids':'spectrogram_id','patient_ids':'patient_id'})
    master_label            = master_label.merge(master_meta[['label_id','fold']],on='label_id',how='left')
    uni_eeg_id              = list(master_label['eeg_id'].unique())
    uni_label_id            = list(master_label['label_id'].unique())
    uni_eeg_id.sort()
    uni_label_id.sort()
    master_label            = master_label.sort_values(by=['patient_id', 'eeg_id'], ascending=[True, True]).reset_index(drop=True)
    master_label.to_csv(f'{save_output_dir}/master_label.csv',index=False)
    
elif label_mode == 'center':
    master_label_all        = pd.read_csv(f'{output_dir}/{dir_master}/df_oof_labels_all.csv')
    master_meta             = pd.read_csv(f'{output_dir}/{dir_master}/train_meta.csv')
    master_label_all        = master_label_all.merge(master_meta[['label_id','flg1_vote']],on='label_id',how='left')
    #各eeg_idの中央値を取得する処理
    master_label            = master_label_all.groupby('eeg_id').apply(lambda x: x.iloc[len(x) // 2])
    master_label            = master_label[master_label['flg1_vote']==True].reset_index(drop=True)
    uni_eeg_id              = list(master_label['eeg_id'].unique())
    uni_label_id            = list(master_label['label_id'].unique())
    uni_eeg_id.sort()
    uni_label_id.sort()
    master_label            = master_label.sort_values(by=['patient_id', 'eeg_id'], ascending=[True, True]).reset_index(drop=True)
    master_label            = master_label[col_labels + \
                                            ['label_id','eeg_id','spectrogram_id','patient_id','flg1_vote','fold']]
    master_label.to_csv(f'{save_output_dir}/master_label.csv',index=False)
    
#read&save_oof
exp_ids                     = [exp_id_spec,exp_id_eeg_wave,exp_id_eeg_img,exp_id_multi]
for exp_id_type in exp_ids:
    for exp_id in exp_id_type:
        LOGGER.info(f'==Read exp_id: {exp_id}==')
        if ('result_TH' in exp_id)or('result_KM' in exp_id):
            load_output_dir = f'{output_dir}/{exp_id}'
            df_oof_output   = pd.read_csv(f'{load_output_dir}/df_oof_output.csv')#全label_idデータ
            col_select      = ['label_id','eeg_id','spectrogram_id','patient_id','fold',
                               'logit_seizure_vote','logit_lpd_vote','logit_gpd_vote','logit_lrda_vote','logit_grda_vote','logit_other_vote',
                                ]
            df_oof_output   = df_oof_output[col_select]
            df_oof_output   = df_oof_output.rename(columns={'logit_seizure_vote':'seizure_logits',
                                                            'logit_lpd_vote':'lpd_logits',
                                                            'logit_gpd_vote':'gpd_logits',
                                                            'logit_lrda_vote':'lrda_logits',
                                                            'logit_grda_vote':'grda_logits',
                                                            'logit_other_vote':'other_logits',
                                                           })
            df_oof_output[col_labels]   = softmax_second_axis(df_oof_output[col_labels_logits].values)
            df_oof_output               = df_oof_output[df_oof_output['label_id'].isin(uni_label_id)].reset_index(drop=True)
        else:
            if label_mode == 'specified':
                df_oof_output,_,_       = read_result(exp_id,col_labels,output_dir,use_tta=False,print_data=False,all_mode=all_mode)
            else:
                load_output_dir         = f'{output_dir}/{exp_id}'
                df_oof_output_all       = pd.read_csv(f'{load_output_dir}/df_oof_output_all.csv')#全label_idデータ
                df_oof_output_all[col_labels]   = softmax_second_axis(df_oof_output_all[col_labels_logits].values)
                df_oof_output           = df_oof_output_all[df_oof_output_all['label_id'].isin(uni_label_id)].reset_index(drop=True)
        #select
        if select_id=='eeg_id':
            df_oof_output               = df_oof_output[df_oof_output['eeg_id'].isin(uni_eeg_id)]
        elif select_id=='label_id':
            df_oof_output               = df_oof_output[df_oof_output['label_id'].isin(uni_label_id)]
        df_oof_output = df_oof_output.sort_values(by=['patient_id', 'eeg_id'], ascending=[True, True]).reset_index(drop=True)
        #Error
        if len(df_oof_output) != len(master_label):
            LOGGER.info(f'len(df_oof_output): {len(df_oof_output)}')
            LOGGER.info(f'len(master_label): {len(master_label)}')
            LOGGER.info('Error')
            aa
        df_oof_output.to_csv(f'{save_output_dir}/df_oof_{exp_id}.csv',index=False)  


# # 2_Stacking

# In[5]:


import itertools
import trainer.train_pp
importlib.reload(trainer.train_pp)
from trainer.train_pp import train_loop_pp
#dir
os.makedirs(f'{save_output_dir}/stacking',exist_ok=True)
save_output_dir_stacking        = f'{save_output_dir}/stacking'

if calc_stack:
    #setting
    if stacking_mode=='all_comb':
        comb4                   = list(itertools.product(exp_id_spec, exp_id_eeg_wave, exp_id_eeg_img, exp_id_multi))
        list_comb4              = [list(comb) for comb in comb4]
        comb3                   = list(itertools.product(exp_id_spec, exp_id_eeg_wave, exp_id_eeg_img))
        list_comb3              = [list(comb) for comb in comb3]
        list_comb               = list_comb4
        
    elif stacking_mode=='set_comb':
        list_comb               = config[exp_id_fold]['set_comb']

    #stacking
    base_cols                   = ['eeg_id', 'spectrogram_id','patient_id','label_id', 'fold']

    id_stacking                 = []
    id_comb                     = []
    metrics_stacking            = []
    for idx_stack,comb in enumerate(list_comb):
        os.makedirs(f'{save_output_dir}/stacking/{idx_stack}',exist_ok=True)
        save_output_dir_stacking_idx        = f'{save_output_dir}/stacking/{idx_stack}'
        id_stacking.append(f'stacking_{idx_stack}')
        id_comb.append(comb)
        #=get_feat=#
        col_labels_allfeat          = []
        for idx_exp,exp_id in enumerate(comb):
            LOGGER.info(f'==Read exp_id: {exp_id}==')
            df_oof_output           = pd.read_csv(f'{save_output_dir}/df_oof_{exp_id}.csv')
            df_oof_output           = df_oof_output[base_cols+col_labels_feat]
            df_oof_output           = df_oof_output.sort_values(by=['patient_id', 'eeg_id'], ascending=[True, True]).reset_index(drop=True)
            #rename
            column_mapping          = {col: f"{col}_{idx_exp}" for col in col_labels_feat if col in df_oof_output.columns}
            new_columns             = list(column_mapping.values())
            df_oof_output           = df_oof_output.rename(columns=column_mapping)
            col_labels_allfeat      +=new_columns
            if idx_exp==0:
                df_oof_output_all   = df_oof_output
            else:
                df_oof_output_all   = df_oof_output_all.merge(df_oof_output[['patient_id','eeg_id']+new_columns],on=['patient_id','eeg_id'],how='left')
        #=add_label=#
        df_oof_output_all           = df_oof_output_all.merge(master_label[['patient_id','eeg_id']+col_labels],on=['patient_id','eeg_id'],how='left')
        
        #stacking
        LOGGER.info(f'==Stacking_{idx_stack}: {comb}==')
        df_oof_output_stacking,oof_metrics_stacking     = train_loop_pp(df_oof_output_all,col_labels_allfeat,col_labels,config,
                                                                        save_output_dir_stacking_idx,LOGGER,
                                                                        mode='stacking')
        df_oof_output_stacking                          = df_oof_output_stacking.rename(columns={'label_ids':'label_id','eeg_ids':'eeg_id','spectrogram_ids':'spectrogram_id','patient_ids':'patient_id'})
        df_oof_output_stacking                          = df_oof_output_stacking.sort_values(by=['patient_id', 'eeg_id'], ascending=[True, True]).reset_index(drop=True)
        df_oof_output_stacking.to_csv(f'{save_output_dir_stacking}/df_oof_stacking_{idx_stack}.csv',index=False)
        metrics_stacking.append(oof_metrics_stacking)
    #save_stacking_list
    df_stacking                     = pd.DataFrame({'id_stacking':id_stacking,'id_comb':id_comb,'metrics':metrics_stacking})
    df_stacking.to_csv(f'{save_output_dir_stacking}/df_stacking.csv',index=False)
else:
    pass


# # 3 Ens1 

# In[ ]:


import postprocess.utils_read_stack_pp_ens
importlib.reload(postprocess.utils_read_stack_pp_ens)
from postprocess.utils_read_stack_pp_ens import preprocess_ensemble,calc_optuna

#dir
os.makedirs(f'{save_output_dir}/ens1',exist_ok=True)
save_output_dir_ens1    = f'{save_output_dir}/ens1'
#setting
calc_init               = True

#=preprocess_ens1_spec=#
weight_init_spec_tmp    = [1]*len(exp_id_spec)
model_type_spec         = ['SPEC']*len(exp_id_spec)
path_spec               = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_spec]
weight_init_spec,oof_data_spec,label_data_spec,criterion_spec,df_oof_spec       = preprocess_ensemble(exp_id_spec,path_spec,col_labels_feat,weight_init_spec_tmp,
                                                                                            master_label,col_labels,calc_init=calc_init)

#=preprocess_ens1_wave=#
weight_init_wave_tmp    = [2]*len(exp_id_eeg_wave)
model_type_wave         = ['WAVE']*len(exp_id_eeg_wave)
path_wave               = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_eeg_wave]
weight_init_wave,oof_data_wave,label_data_wave,criterion_wave,df_oof_wave       = preprocess_ensemble(exp_id_eeg_wave,path_wave,col_labels_feat,weight_init_wave_tmp,
                                                                                            master_label,col_labels,calc_init=calc_init)

#=preprocess_ens1_img=#
weight_init_img_tmp     = [1]*len(exp_id_eeg_img)
model_type_img          = ['IMG']*len(exp_id_eeg_img)
path_img                = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_eeg_img]
weight_init_img,oof_data_img,label_data_img,criterion_img,df_oof_img            = preprocess_ensemble(exp_id_eeg_img,path_img,col_labels_feat,weight_init_img_tmp,
                                                                                            master_label,col_labels,calc_init=calc_init)

#=preprocess_ens1_multi=#
weight_init_multi_tmp   = [4]*len(exp_id_multi)
model_type_multi        = ['MULTI']*len(exp_id_multi)
path_multi              = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_multi]
weight_init_multi,oof_data_multi,label_data_multi,criterion_multi,df_oof_multi  = preprocess_ensemble(exp_id_multi,path_multi,col_labels_feat,weight_init_multi_tmp,
                                                                                            master_label,col_labels,calc_init=calc_init)

#=preprocess_ens1_stack=#
if calc_stack:
    exp_id_stack            = df_stacking['id_stacking'].values.tolist()
    weight_init_stack_tmp   = [6]*len(exp_id_stack)
    model_type_stack        = ['STACK']*len(exp_id_stack)
    path_stack              = [f'{save_output_dir_stacking}/df_oof_stacking_{idx_stack}.csv' for idx_stack in range(len(exp_id_stack))]
    weight_init_stack,oof_data_stack,label_data_stack,criterion_stack,df_oof_stack  = preprocess_ensemble(exp_id_stack,path_stack,col_labels_feat,weight_init_stack_tmp,
                                                                                                master_label,col_labels,calc_init=calc_init)
else:
    exp_id_stack            = []
    weight_init_stack_tmp   = []
    model_type_stack        = []
    path_stack              = []
    weight_init_stack       = []
    oof_data_stack          = []
    label_data_stack        = []
    criterion_stack         = []
    df_oof_stack            = []
    
#=preprocess_ens1_all=#
exp_id_ens1_all             = exp_id_spec + exp_id_eeg_wave + exp_id_eeg_img + exp_id_multi \
                              + exp_id_stack
weight_init_ens1_all_tmp    = weight_init_spec_tmp + weight_init_wave_tmp + weight_init_img_tmp + weight_init_multi_tmp \
                              + weight_init_stack_tmp
model_type_ens1_all         = model_type_spec + model_type_wave + model_type_img + model_type_multi\
                              + model_type_stack
path_ens1_all               = path_spec + path_wave + path_img + path_multi \
                              +path_stack
weight_init_ens1_all,oof_data_ens1_all,label_data_ens1_all,criterion_ens1_all,df_oof_ens1_all \
                                                                                = preprocess_ensemble(exp_id_ens1_all,path_ens1_all,col_labels_feat,weight_init_ens1_all_tmp,
                                                                                            master_label,col_labels,calc_init=calc_init)


# In[ ]:


print('=========OPTUNA=========')
#spec
print('===spec===')
df_ens1_spec,oof_data_ens1_spec_logits,oof_data_ens1_spec_proba         = calc_optuna(exp_id_spec,weight_init_spec,oof_data_spec,
                                                                    label_data_spec,criterion_spec,n_trials=config['optuna']['n_trials_spec'])
df_ens1_spec['model_type']              = model_type_spec
df_oof_ens1_spec                        = df_oof_spec.copy()
df_oof_ens1_spec[col_labels_feat]       = oof_data_ens1_spec_logits
df_oof_ens1_spec[col_labels]            = oof_data_ens1_spec_proba
df_ens1_spec.to_csv(f'{save_output_dir_ens1}/df_ens1_spec.csv',index=False)
df_oof_ens1_spec.to_csv(f'{save_output_dir_ens1}/df_oof_spec.csv',index=False)

#eeg_wave
print('===eeg_wave===')
df_ens1_wave,oof_data_ens1_wave_logits,oof_data_ens1_wave_proba         = calc_optuna(exp_id_eeg_wave,weight_init_wave,oof_data_wave,
                                                                    label_data_wave,criterion_wave,n_trials=config['optuna']['n_trials_wave'])
df_ens1_wave['model_type']              = model_type_wave
df_oof_ens1_wave                        = df_oof_wave.copy()
df_oof_ens1_wave[col_labels_feat]       = oof_data_ens1_wave_logits
df_oof_ens1_wave[col_labels]            = oof_data_ens1_wave_proba
df_ens1_wave.to_csv(f'{save_output_dir_ens1}/df_ens1_wave.csv',index=False)
df_oof_ens1_wave.to_csv(f'{save_output_dir_ens1}/df_oof_wave.csv',index=False)

#eeg_img
print('===eeg_img===')
df_ens1_img,oof_data_ens1_img_logits,oof_data_ens1_img_proba           = calc_optuna(exp_id_eeg_img,weight_init_img,oof_data_img,
                                                                    label_data_img,criterion_img,n_trials=config['optuna']['n_trials_img'])
df_ens1_img['model_type']               = model_type_img
df_oof_ens1_img                         = df_oof_img.copy()
df_oof_ens1_img[col_labels_feat]        = oof_data_ens1_img_logits
df_oof_ens1_img[col_labels]             = oof_data_ens1_img_proba
df_ens1_img.to_csv(f'{save_output_dir_ens1}/df_ens1_img.csv',index=False)
df_oof_ens1_img.to_csv(f'{save_output_dir_ens1}/df_oof_img.csv',index=False)

#multi
print('===multi===')
df_ens1_multi,oof_data_ens1_multi_logits,oof_data_ens1_multi_proba       = calc_optuna(exp_id_multi,weight_init_multi,oof_data_multi,
                                                                    label_data_multi,criterion_multi,n_trials=config['optuna']['n_trials_multi'])
df_ens1_multi['model_type']             = model_type_multi
df_oof_ens1_multi                       = df_oof_multi.copy()
df_oof_ens1_multi[col_labels_feat]      = oof_data_ens1_multi_logits
df_oof_ens1_multi[col_labels]           = oof_data_ens1_multi_proba
df_ens1_multi.to_csv(f'{save_output_dir_ens1}/df_ens1_multi.csv',index=False)
df_oof_ens1_multi.to_csv(f'{save_output_dir_ens1}/df_oof_multi.csv',index=False)

#stack
if calc_stack:
    print('===stack===')
    df_ens1_stack,oof_data_ens1_stack_logits,oof_data_ens1_stack_proba       = calc_optuna(exp_id_stack,weight_init_stack,oof_data_stack,
                                                                        label_data_stack,criterion_stack,n_trials=config['optuna']['n_trials_stack'])
    df_ens1_stack['model_type']             = model_type_stack
    df_oof_ens1_stack                       = df_oof_stack.copy()
    df_oof_ens1_stack[col_labels_feat]      = oof_data_ens1_stack_logits
    df_oof_ens1_stack[col_labels]           = oof_data_ens1_stack_proba
    df_ens1_stack.to_csv(f'{save_output_dir_ens1}/df_ens1_stack.csv',index=False)
    df_oof_ens1_stack.to_csv(f'{save_output_dir_ens1}/df_oof_stack.csv',index=False)
else:
    pass
#all
print('===all===')
df_ens1_all,oof_data_ens1_all_logits,oof_data_ens1_all_proba           = calc_optuna(exp_id_ens1_all,weight_init_ens1_all,oof_data_ens1_all,
                                                                    label_data_ens1_all,criterion_ens1_all,n_trials=config['optuna']['n_trials_ens1'])
df_ens1_all['model_type']               = model_type_ens1_all
df_oof_ens1_all[col_labels_feat]        = oof_data_ens1_all_logits
df_oof_ens1_all[col_labels]             = oof_data_ens1_all_proba
df_ens1_all.to_csv(f'{save_output_dir_ens1}/df_ens1_all.csv',index=False)
df_oof_ens1_all.to_csv(f'{save_output_dir_ens1}/df_oof_ens1_all.csv',index=False)


# # 4_PP

# In[ ]:


import trainer.train_pp
importlib.reload(trainer.train_pp)
from trainer.train_pp import preprocess_pp

if calc_pp:
    #dir
    os.makedirs(f'{save_output_dir}/pp',exist_ok=True)
    save_output_dir_pp              = f'{save_output_dir}/pp'

    #get_feat
    dict_df_oof_ens1                = {}
    dict_df_oof_ens1['spec']        = df_oof_ens1_spec
    dict_df_oof_ens1['wave']        = df_oof_ens1_wave
    dict_df_oof_ens1['img']         = df_oof_ens1_img
    dict_df_oof_ens1['multi']       = df_oof_ens1_multi
    dict_df_oof_ens1['stack']       = df_oof_ens1_stack
    dict_df_oof_ens1['ens']         = df_oof_ens1_all

    #settings
    id_pp                           = []
    id_setting_pp                   = []
    metrics_pp                      = []
    num_set_pp                      = config[exp_id_fold]['set_pp']['num_set']
    for idx_pp in range(num_set_pp):
        os.makedirs(f'{save_output_dir}/pp/{idx_pp}',exist_ok=True)
        save_output_dir_pp_idx        = f'{save_output_dir}/pp/{idx_pp}'
        setting_pp                  = config[exp_id_fold]['set_pp'][f'set{idx_pp}']
        id_pp.append(f'pp_{idx_pp}')
        id_setting_pp.append(setting_pp)

        df_feat,new_col_labels_all              = preprocess_pp(dict_df_oof_ens1,setting_pp,col_labels_feat,
                                                    master_label,col_labels,)
        
        LOGGER.info(f'==PP_{idx_pp}==')
        LOGGER.info(f'==NUM_FEAT_{idx_pp}: {len(new_col_labels_all)}==')
        df_oof_output_pp,oof_metrics_pp         = train_loop_pp(df_feat,new_col_labels_all ,col_labels,config,
                                                        save_output_dir_pp_idx,LOGGER,
                                                        mode='pp')
        df_oof_output_pp                        = df_oof_output_pp.rename(columns={'label_ids':'label_id','eeg_ids':'eeg_id','spectrogram_ids':'spectrogram_id','patient_ids':'patient_id'})
        df_oof_output_pp                        = df_oof_output_pp.sort_values(by=['patient_id', 'eeg_id'], ascending=[True, True]).reset_index(drop=True)
        df_oof_output_pp.to_csv(f'{save_output_dir_pp}/df_oof_pp_{idx_pp}.csv',index=False)
        metrics_pp.append(oof_metrics_pp)

    #save_pp_list
    df_pp                           = pd.DataFrame({'id_pp':id_pp,'id_setting_pp':id_setting_pp,'metrics':metrics_pp})
    df_pp.to_csv(f'{save_output_dir_pp}/df_pp.csv',index=False)


# # 5 Ens2 

# In[ ]:


if calc_pp:
    import postprocess.utils_read_stack_pp_ens
    importlib.reload(postprocess.utils_read_stack_pp_ens)
    from postprocess.utils_read_stack_pp_ens import preprocess_ensemble,calc_optuna

    #dir
    os.makedirs(f'{save_output_dir}/ens2',exist_ok=True)
    save_output_dir_ens2    = f'{save_output_dir}/ens2'

    #setting
    calc_init               = True

    #=preprocess_ens2_spec=#
    weight_init_spec_tmp    = [1]*len(exp_id_spec)
    model_type_spec         = ['SPEC']*len(exp_id_spec)
    path_spec               = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_spec]

    #=preprocess_ens2_wave=#
    weight_init_wave_tmp    = [2]*len(exp_id_eeg_wave)
    model_type_wave         = ['WAVE']*len(exp_id_eeg_wave)
    path_wave               = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_eeg_wave]

    #=preprocess_ens2_img=#
    weight_init_img_tmp     = [1]*len(exp_id_eeg_img)
    model_type_img          = ['IMG']*len(exp_id_eeg_img)
    path_img                = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_eeg_img]

    #=preprocess_ens2_multi=#
    weight_init_multi_tmp   = [4]*len(exp_id_multi)
    model_type_multi        = ['MULTI']*len(exp_id_multi)
    path_multi              = [f'{save_output_dir}/df_oof_{exp_id}.csv' for exp_id in exp_id_multi]

    #=preprocess_ens2_stack=#
    exp_id_stack            = df_stacking['id_stacking'].values.tolist()
    weight_init_stack_tmp   = [6]*len(exp_id_stack)
    model_type_stack        = ['STACK']*len(exp_id_stack)
    path_stack              = [f'{save_output_dir_stacking}/df_oof_stacking_{idx_stack}.csv' for idx_stack in range(len(exp_id_stack))]

    #=preprocess_ens2_pp=#
    exp_id_pp               = df_pp['id_pp'].values.tolist()
    weight_init_pp_tmp      = [20]*len(exp_id_pp)
    model_type_pp           = ['PP']*len(exp_id_pp)
    path_pp                 = [f'{save_output_dir_pp}/df_oof_pp_{idx_pp}.csv' for idx_pp in range(len(exp_id_pp))]

    #=preprocess_ens2_all=#
    exp_id_ens2_all             = exp_id_spec + exp_id_eeg_wave + exp_id_eeg_img + exp_id_multi \
                                + exp_id_stack + exp_id_pp
    weight_init_ens2_all_tmp    = weight_init_spec_tmp + weight_init_wave_tmp + weight_init_img_tmp + weight_init_multi_tmp \
                                + weight_init_stack_tmp + weight_init_pp_tmp
    model_type_ens2_all         = model_type_spec + model_type_wave + model_type_img + model_type_multi\
                                + model_type_stack + model_type_pp
    path_ens2_all               = path_spec + path_wave + path_img + path_multi \
                                +path_stack + path_pp

    weight_init_ens2_all,oof_data_ens2_all,label_data_ens2_all,criterion_ens2_all,df_oof_ens2_all \
                                                                                    = preprocess_ensemble(exp_id_ens2_all,path_ens2_all,col_labels_feat,weight_init_ens2_all_tmp,
                                                                                                master_label,col_labels,calc_init=calc_init)


    print('=========OPTUNA=========')
    print('===all===')
    df_ens2_all,oof_data_ens2_all_logits,oof_data_ens2_all      = calc_optuna(exp_id_ens2_all,weight_init_ens2_all,oof_data_ens2_all,
                                                                    label_data_ens2_all,criterion_ens2_all,n_trials=config['optuna']['n_trials_ens2'])
    df_ens2_all['model_type']               = model_type_ens2_all
    df_oof_ens2_all[col_labels_feat]        = oof_data_ens2_all_logits
    df_oof_ens2_all[col_labels]             = oof_data_ens2_all
    df_ens2_all.to_csv(f'{save_output_dir_ens2}/df_ens2_all.csv',index=False)
    df_oof_ens2_all.to_csv(f'{save_output_dir_ens2}/df_oof_ens2_all.csv',index=False)

