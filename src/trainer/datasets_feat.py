# ===============
# Import
# ===============
import numpy as np
import librosa

###################
# select_eef_data
###################
def select_eeg_data(sub_eeg_data,wave_select,
                    dict_col_index):
    if wave_select=='delta':
        #LL
        sub_eeg_data_Fp1_F7         = sub_eeg_data[:,dict_col_index['ind_eeg_Fp1']] - sub_eeg_data[:,dict_col_index['ind_eeg_F7']]
        sub_eeg_data_F7_T3          = sub_eeg_data[:,dict_col_index['ind_eeg_F7']] - sub_eeg_data[:,dict_col_index['ind_eeg_T3']]
        sub_eeg_data_T3_T5          = sub_eeg_data[:,dict_col_index['ind_eeg_T3']] - sub_eeg_data[:,dict_col_index['ind_eeg_T5']]
        sub_eeg_data_T5_O1          = sub_eeg_data[:,dict_col_index['ind_eeg_T5']] - sub_eeg_data[:,dict_col_index['ind_eeg_O1']]
        #LP
        sub_eeg_data_Fp1_F3         = sub_eeg_data[:,dict_col_index['ind_eeg_Fp1']] - sub_eeg_data[:,dict_col_index['ind_eeg_F3']]
        sub_eeg_data_F3_C3          = sub_eeg_data[:,dict_col_index['ind_eeg_F3']] - sub_eeg_data[:,dict_col_index['ind_eeg_C3']]
        sub_eeg_data_C3_P3          = sub_eeg_data[:,dict_col_index['ind_eeg_C3']] - sub_eeg_data[:,dict_col_index['ind_eeg_P3']]
        sub_eeg_data_P3_O1          = sub_eeg_data[:,dict_col_index['ind_eeg_P3']] - sub_eeg_data[:,dict_col_index['ind_eeg_O1']]
        #RP
        sub_eeg_data_Fp2_F4         = sub_eeg_data[:,dict_col_index['ind_eeg_Fp2']] - sub_eeg_data[:,dict_col_index['ind_eeg_F4']]
        sub_eeg_data_F4_C4          = sub_eeg_data[:,dict_col_index['ind_eeg_F4']] - sub_eeg_data[:,dict_col_index['ind_eeg_C4']]
        sub_eeg_data_C4_P4          = sub_eeg_data[:,dict_col_index['ind_eeg_C4']] - sub_eeg_data[:,dict_col_index['ind_eeg_P4']]
        sub_eeg_data_P4_O2          = sub_eeg_data[:,dict_col_index['ind_eeg_P4']] - sub_eeg_data[:,dict_col_index['ind_eeg_O2']]
        #RL
        sub_eeg_data_Fp2_F8         = sub_eeg_data[:,dict_col_index['ind_eeg_Fp2']] - sub_eeg_data[:,dict_col_index['ind_eeg_F8']]
        sub_eeg_data_F8_T4          = sub_eeg_data[:,dict_col_index['ind_eeg_F8']] - sub_eeg_data[:,dict_col_index['ind_eeg_T4']]
        sub_eeg_data_T4_T6          = sub_eeg_data[:,dict_col_index['ind_eeg_T4']] - sub_eeg_data[:,dict_col_index['ind_eeg_T6']]
        sub_eeg_data_T6_O2          = sub_eeg_data[:,dict_col_index['ind_eeg_T6']] - sub_eeg_data[:,dict_col_index['ind_eeg_O2']]
        #===stack===#
        sub_eeg_data_wave           = np.stack([sub_eeg_data_Fp1_F7,sub_eeg_data_F7_T3,sub_eeg_data_T3_T5,sub_eeg_data_T5_O1,#LL
                                                sub_eeg_data_Fp1_F3,sub_eeg_data_F3_C3,sub_eeg_data_C3_P3,sub_eeg_data_P3_O1,#LP
                                                sub_eeg_data_Fp2_F4,sub_eeg_data_F4_C4,sub_eeg_data_C4_P4,sub_eeg_data_P4_O2,#RP
                                                sub_eeg_data_Fp2_F8,sub_eeg_data_F8_T4,sub_eeg_data_T4_T6,sub_eeg_data_T6_O2,#RL
                                                ],axis=0).squeeze() #(ch,time)=(19,10000)
        sub_eeg_data_wave           = np.clip(sub_eeg_data_wave,-10000,10000) 
        
    return sub_eeg_data_wave

# ==============================
# get_eeg_img_fea
# ==============================
def get_eeg_img_feat(sub_eeg_data_wave_raw,sr_eeg_resample,config,
                        calc_img1_stft):
    #====IMG1 STFT====#
    if calc_img1_stft:
        #setting
        img1_res_type_eeg           = "kaiser_fast"
        img1_N_fft_eeg              = config['dataset']['EEG_IMG']['IMG1_stft']['N_fft_eeg']
        img1_hop_eeg                = config['dataset']['EEG_IMG']['IMG1_stft']['hop_eeg']
        img1_max_freq               = config['dataset']['EEG_IMG']['IMG1_stft']['max_freq']
        img1_max                    = config['dataset']['EEG_IMG']['IMG1_stft']['max']
        img1_min                    = config['dataset']['EEG_IMG']['IMG1_stft']['min']
        img1_clip_coeff             = config['dataset']['EEG_IMG']['IMG1_stft']['clip_coeff']
        #calc
        sub_eeg_data_img_1          = []
        freq_frame_eeg_img          = librosa.fft_frequencies(sr=sr_eeg_resample, n_fft=img1_N_fft_eeg)
        delt_freq                   = freq_frame_eeg_img[1] - freq_frame_eeg_img[0]
        max_freq                    = img1_max_freq #[Hz] 20Hzまで
        num_select_freq             = int(max_freq/delt_freq)
        freq_frame_eeg_img          = freq_frame_eeg_img[:num_select_freq]
        for ch in range(sub_eeg_data_wave_raw.shape[0]):
            y                       = sub_eeg_data_wave_raw[ch,:]# チャンネルデータを抽出             
            stft                    = librosa.stft(y=y,n_fft=img1_N_fft_eeg, hop_length=img1_hop_eeg)# スペクトログラムの計算 (freq,time)
            stft                    = np.abs(stft[:num_select_freq,:]) #select_freq 
            stft                    = np.log(stft+1e-6)#amplitude_to_db
            sub_eeg_data_img_1.append(stft)
        sub_eeg_data_img_1          = np.array(sub_eeg_data_img_1)
        #===norm_img===#
        calc_norm_img               = False
        if calc_norm_img:#各waveファイルごとに規格化
            ep                      = 1e-6
            max                     = np.nanmean(sub_eeg_data_img_1.flatten())
            std                     = np.nanstd(sub_eeg_data_img_1.flatten())
            sub_eeg_data_img_1      = (sub_eeg_data_img_1-max)/(std+ep)
            sub_eeg_data_img_1      = np.nan_to_num(sub_eeg_data_img_1, nan=0.0)
        else:#全データ一律規格化
            ep                      = 1e-6
            max                     = img1_max 
            min                     = img1_min
            sub_eeg_data_img_1      = np.clip(sub_eeg_data_img_1,img1_clip_coeff*min,img1_clip_coeff*max) 
            sub_eeg_data_img_1      = (sub_eeg_data_img_1-0.5*(max+min+ep))/(max-min+ep)
            sub_eeg_data_img_1      = np.nan_to_num(sub_eeg_data_img_1, nan=0.0)
    else:
        sub_eeg_data_img_1          = []    


    return sub_eeg_data_img_1

