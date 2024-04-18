
import pandas as pd
import numpy as np
import polars as pl
import importlib
import sys

sys.path.append("..")
from utils.utils import pickle_dump,pickle_load,seed_everything,AttrDict,replace_placeholders
from utils.logger import setup_logger, LOGGER


####################################################################################################
def calc_numdata(train_meta,files_train_eegs,files_train_spectrograms,
                 files_train_eegs_names,files_train_spectrograms_names):
    print('train_data:      ', len(train_meta))
    print('eeg_id:          ', train_meta['eeg_id'].nunique())
    print('eeg_file:        ', len(files_train_eegs))
    print('spectrogram_id:  ', train_meta['spectrogram_id'].nunique())
    print('spectrogram_file:', len(files_train_spectrograms))
    print('label_id:        ', train_meta['label_id'].nunique())
    print('patient_id:      ', train_meta['patient_id'].nunique())
    matching_row_eegs               = train_meta[train_meta['eeg_id'].isin(files_train_eegs_names)]
    matching_row_spectrograms       = train_meta[train_meta['spectrogram_id'].isin(files_train_spectrograms_names)]
    print('len_matching_row_eegs:           ', len(matching_row_eegs))
    print('len_matching_row_spectrograms:   ', len(matching_row_spectrograms))

####################################################################################################
def read_rawdata(config,data_dir,output_dir,input_type,
                 files_train_eegs_names,files_train_spectrograms_names,
                 phase='train'):
    data_read_mode                              = config['dataset']['data_read_mode']
    #==init==#
    train_eegs_data_dict                    = {}
    len_train_eegs_data_dict                = {}
    train_spectrograms_data_dict            = {}
    len_train_spectrograms_data_dict        = {}
    sr_train_spectrograms_data_dict         = {}
    col_eegs                                = []
    col_spectrograms                        = []
    dict_col_index                          = dict()
    list_nan_eeg_id                         = []
    list_nan_spec_id                        = []
    list_max_eeg                            = []
    list_min_eeg                            = []
    if data_read_mode != 'load':
        if data_read_mode == 'pass':
            pass
        else:
            #==eegs==#
            if (input_type=='SPEC')or(input_type=='EEG_WAVE') or (input_type=='EEG_IMG') or (input_type=='ALL'):
                #sample_rate=5[ms] 200[Hz]  #sample_rate=2[s]
                train_eegs_data_dict                    = {}
                len_train_eegs_data_dict                = {}
                for idx,eeg_id in enumerate(files_train_eegs_names):
                    if idx==0:
                        col_eegs                        = pl.read_parquet(f'{data_dir}/{phase}_eegs/{eeg_id}.parquet').columns
                    data_np                             = pl.read_parquet(f'{data_dir}/{phase}_eegs/{eeg_id}.parquet').to_numpy()
                    if (np.isnan(data_np.max()))and(phase!='test'):#eegはtestデータ中にnanはないらしい(probingにより)ので削除する。
                        list_nan_eeg_id.append(eeg_id)
                        #nan対策 ch毎の平均埋め
                        for id_ch in range(data_np.shape[1]):
                            x                   = data_np[:, id_ch]  # convert to float32
                            mean                = np.nanmean(x)  # arithmetic mean along the specified axis, ignoring NaNs
                            nan_percentage      = np.isnan(x).mean()  # percentage of NaN values in feature
                            # === Fill nan values ===
                            if nan_percentage < 1:  # if some values are nan, but not all
                                x               = np.nan_to_num(x, nan=mean)
                            else:  # if all values are nan
                                x[:]            = 0
                            data_np[:, id_ch]   =  x
                    else:
                        list_max_eeg.append(data_np.max())
                        list_min_eeg.append(data_np.min())
                    train_eegs_data_dict[eeg_id]        = data_np
                len_train_eegs_data_list                = list(len_train_eegs_data_dict.values())#各データサイズ→バラバラ   10000～684400=50[s]～3422[s] 大体は50[s]
                list_max_eeg                            = np.array(list_max_eeg)
                list_min_eeg                            = np.array(list_min_eeg)
                dict_col_index['ind_eeg_Fp1']       = [i for i, col in enumerate(col_eegs) if col.startswith('Fp1')]
                dict_col_index['ind_eeg_Fp2']       = [i for i, col in enumerate(col_eegs) if col.startswith('Fp2')]
                dict_col_index['ind_eeg_F3']        = [i for i, col in enumerate(col_eegs) if col.startswith('F3')]
                dict_col_index['ind_eeg_F4']        = [i for i, col in enumerate(col_eegs) if col.startswith('F4')]
                dict_col_index['ind_eeg_F7']        = [i for i, col in enumerate(col_eegs) if col.startswith('F7')]
                dict_col_index['ind_eeg_F8']        = [i for i, col in enumerate(col_eegs) if col.startswith('F8')]
                dict_col_index['ind_eeg_C3']        = [i for i, col in enumerate(col_eegs) if col.startswith('C3')]
                dict_col_index['ind_eeg_C4']        = [i for i, col in enumerate(col_eegs) if col.startswith('C4')]
                dict_col_index['ind_eeg_T3']        = [i for i, col in enumerate(col_eegs) if col.startswith('T3')]
                dict_col_index['ind_eeg_T4']        = [i for i, col in enumerate(col_eegs) if col.startswith('T4')]
                dict_col_index['ind_eeg_T5']        = [i for i, col in enumerate(col_eegs) if col.startswith('T5')]
                dict_col_index['ind_eeg_T6']        = [i for i, col in enumerate(col_eegs) if col.startswith('T6')]
                dict_col_index['ind_eeg_P3']        = [i for i, col in enumerate(col_eegs) if col.startswith('P3')]
                dict_col_index['ind_eeg_P4']        = [i for i, col in enumerate(col_eegs) if col.startswith('P4')]
                dict_col_index['ind_eeg_O1']        = [i for i, col in enumerate(col_eegs) if col.startswith('O1')]
                dict_col_index['ind_eeg_O2']        = [i for i, col in enumerate(col_eegs) if col.startswith('O2')]
                dict_col_index['ind_eeg_Fz']        = [i for i, col in enumerate(col_eegs) if col.startswith('Fz')]
                dict_col_index['ind_eeg_Cz']        = [i for i, col in enumerate(col_eegs) if col.startswith('Cz')]
                dict_col_index['ind_eeg_Pz']        = [i for i, col in enumerate(col_eegs) if col.startswith('Pz')]
                dict_col_index['ind_eeg_EKG']       = [i for i, col in enumerate(col_eegs) if col.startswith('EKG')]
                if data_read_mode=='save':
                    pickle_dump(train_eegs_data_dict, f'{output_dir}/data_raw/train_eegs_data_dict.pkl')
                    pickle_dump(len_train_eegs_data_dict, f'{output_dir}/data_raw/len_train_eegs_data_dict.pkl')
                    pickle_dump(col_eegs, f'{output_dir}/data_raw/col_eegs.pkl')
                    pickle_dump(dict_col_index, f'{output_dir}/data_raw/dict_col_index.pkl')
                    pickle_dump(list_max_eeg, f'{output_dir}/data_raw/list_max_eeg.pkl')
                    pickle_dump(list_min_eeg, f'{output_dir}/data_raw/list_min_eeg.pkl')
                    pickle_dump(list_nan_eeg_id, f'{output_dir}/data_raw/list_nan_eeg_id.pkl')

            if (input_type=='SPEC')or(input_type=='EEG_WAVE') or (input_type=='EEG_IMG') or (input_type=='ALL'):
                #==spectrogram==#
                for idx,spectrogram_id in enumerate(files_train_spectrograms_names):
                    if idx==0:
                        col_spectrograms                                = pl.read_parquet(f'{data_dir}/{phase}_spectrograms/{spectrogram_id}.parquet').columns
                    data_np                                             = pl.read_parquet(f'{data_dir}/{phase}_spectrograms/{spectrogram_id}.parquet').to_numpy()
                    if np.isnan(data_np.max()):#
                        for id_ch in range(4):
                            start_ch            = 1+id_ch*100
                            x                   = data_np[:, start_ch:start_ch+100]  # convert to float32
                            mean                = np.nanmean(x)  # arithmetic mean along the specified axis, ignoring NaNs
                            nan_percentage      = np.isnan(x).mean()  # percentage of NaN values in feature
                            # === Fill nan values ===
                            if nan_percentage < 1:  # if some values are nan, but not all
                                x               = np.nan_to_num(x, nan=mean)
                            else:  # if all values are nan
                                x[:,:]          = 0
                            data_np[:, start_ch:start_ch+100]   =  x
                    train_spectrograms_data_dict[spectrogram_id]        = data_np
                    len_train_spectrograms_data_dict[spectrogram_id]    = len(train_spectrograms_data_dict[spectrogram_id])
                    sr_train_spectrograms_data_dict[spectrogram_id]     = train_spectrograms_data_dict[spectrogram_id][1,0]-train_spectrograms_data_dict[spectrogram_id][0,0]
                
                #====#
                dict_col_index['ind_spec_time']     = [0]
                dict_col_index['ind_spec_LL']       = [i for i, col in enumerate(col_spectrograms) if col.startswith('LL')]
                dict_col_index['ind_spec_RL']       = [i for i, col in enumerate(col_spectrograms) if col.startswith('RL')]
                dict_col_index['ind_spec_RP']       = [i for i, col in enumerate(col_spectrograms) if col.startswith('RP')]
                dict_col_index['ind_spec_LP']       = [i for i, col in enumerate(col_spectrograms) if col.startswith('LP')]
                dict_col_index['freq_LL']           = [float(col.split('_')[1]) for i, col in enumerate(col_spectrograms) if col.startswith('LL')]
                dict_col_index['freq_RL']           = [float(col.split('_')[1]) for i, col in enumerate(col_spectrograms) if col.startswith('RL')]
                dict_col_index['freq_RP']           = [float(col.split('_')[1]) for i, col in enumerate(col_spectrograms) if col.startswith('RP')]
                dict_col_index['freq_LP']           = [float(col.split('_')[1]) for i, col in enumerate(col_spectrograms) if col.startswith('LP')]
                dict_col_index['delta_freq_LL']     = [float(dict_col_index['freq_LL'][i+1])-float(dict_col_index['freq_LL'][i]) for i in range(len(dict_col_index['freq_LL'])-1)]
                dict_col_index['delta_freq_RL']     = [float(dict_col_index['freq_RL'][i+1])-float(dict_col_index['freq_RL'][i]) for i in range(len(dict_col_index['freq_RL'])-1)]
                dict_col_index['delta_freq_RP']     = [float(dict_col_index['freq_RP'][i+1])-float(dict_col_index['freq_RP'][i]) for i in range(len(dict_col_index['freq_RP'])-1)]
                dict_col_index['delta_freq_LP']     = [float(dict_col_index['freq_LP'][i+1])-float(dict_col_index['freq_LP'][i]) for i in range(len(dict_col_index['freq_LP'])-1)]
                if data_read_mode=='save':
                    pickle_dump(train_spectrograms_data_dict, f'{output_dir}/data_raw/train_spectrograms_data_dict.pkl')
                    pickle_dump(len_train_spectrograms_data_dict, f'{output_dir}/data_raw/len_train_spectrograms_data_dict.pkl')
                    pickle_dump(col_spectrograms, f'{output_dir}/data_raw/col_spectrograms.pkl')
                    pickle_dump(dict_col_index, f'{output_dir}/data_raw/dict_col_index.pkl')
                    pickle_dump(list_nan_spec_id, f'{output_dir}/data_raw/list_nan_spec_id.pkl')

    elif data_read_mode=='load':
        #==eegs==#
        if (input_type=='SPEC')or(input_type=='EEG_WAVE') or (input_type=='EEG_IMG') or (input_type=='ALL'):
            train_eegs_data_dict                    = pickle_load(f'{output_dir}/data_raw/train_eegs_data_dict.pkl')
            len_train_eegs_data_dict                = pickle_load(f'{output_dir}/data_raw/len_train_eegs_data_dict.pkl')
            col_eegs                                = pickle_load(f'{output_dir}/data_raw/col_eegs.pkl')
            dict_col_index                          = pickle_load(f'{output_dir}/data_raw/dict_col_index.pkl')
            list_max_eeg                            = pickle_load(f'{output_dir}/data_raw/list_max_eeg.pkl')
            list_min_eeg                            = pickle_load(f'{output_dir}/data_raw/list_min_eeg.pkl')
            list_nan_eeg_id                         = pickle_load(f'{output_dir}/data_raw/list_nan_eeg_id.pkl')

        #==spectrogram==#
        if (input_type=='SPEC')or(input_type=='EEG_WAVE') or (input_type=='EEG_IMG') or (input_type=='ALL'):
            train_spectrograms_data_dict            = pickle_load(f'{output_dir}/data_raw/train_spectrograms_data_dict.pkl')
            len_train_spectrograms_data_dict        = pickle_load(f'{output_dir}/data_raw/len_train_spectrograms_data_dict.pkl')
            col_spectrograms                        = pickle_load(f'{output_dir}/data_raw/col_spectrograms.pkl')
            dict_col_index                          = pickle_load(f'{output_dir}/data_raw/dict_col_index.pkl')
            list_nan_spec_id                        = pickle_load(f'{output_dir}/data_raw/list_nan_spec_id.pkl')
    
    return train_eegs_data_dict,len_train_eegs_data_dict,col_eegs,\
            dict_col_index,list_max_eeg,list_min_eeg,list_nan_eeg_id,\
            train_spectrograms_data_dict,len_train_spectrograms_data_dict,\
            col_spectrograms,dict_col_index,list_nan_spec_id

