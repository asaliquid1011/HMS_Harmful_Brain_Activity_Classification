# ===============
# Import
# ===============
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import importlib
# ===============
# import mycode
# ===============
import sys
sys.path.append('../')
import trainer.datasets_aug
importlib.reload(trainer.datasets_aug)
#utils
from audio.audio_aug  import get_wave_transforms,get_spectrogram_transforms
from utils.utils import pickle_dump,pickle_load

from trainer.datasets_aug import augmentation_spec,augmentation_eeg_wave,augmentation_eeg_img
from trainer.datasets_feat import select_eeg_data,get_eeg_img_feat


# ==============================
# Train Dataset
# ==============================
class HMS_Dataset(Dataset):
    def __init__(self,df,eegs_data_dict,spectrograms_data_dict,dict_col_index,
                 config,phase='train'):
        #==other==#
        self.config                 = config
        self.seed                   = config['train']['SEED']
        self.phase                  = phase
        self.epoch                  = 0
        #==mode==#
        self.input_type             = config['dataset']['input_type']
        self.select_data            = config['dataset']['select_data']
        self.Split_Mode             = config['calc_mode']['Split_Mode']
        #==load==#
        self.dataset_mode_EEG_WAVE  = config['dataset']['EEG']['dataset_mode']
        if self.dataset_mode_EEG_WAVE=='load':
            output_dir              = config['path']['output_dir']
            EXP_ID                  = config['EXP_ID']
            save_dataset_path       = f'{output_dir}/{EXP_ID}/dataset/dict_EEG_WAVE.pkl'
            self.dict_EEG_WAVE      = pickle_load(save_dataset_path)

        self.dataset_mode_EEG_IMG   = config['dataset']['EEG_IMG']['dataset_mode']
        dataset_load_EXP_ID         = config['dataset']['EEG_IMG']['dataset_load_EXP_ID']
        if self.dataset_mode_EEG_IMG=='load':
            output_dir              = config['path']['output_dir']
            if dataset_load_EXP_ID=='None':
                EXP_ID              = config['EXP_ID']
            else:
                EXP_ID              = dataset_load_EXP_ID
            save_dataset_path       = f'{output_dir}/{EXP_ID}/dataset/dict_EEG_IMG.pkl'
            self.dict_EEG_IMG       = pickle_load(save_dataset_path)

        #==calc_mode==#
        self.calc_spec              = (self.input_type=='SPEC')or(self.input_type=='ALL')
        self.calc_eeg_wave          = (self.input_type=='EEG_WAVE'and (self.dataset_mode_EEG_WAVE!='load')) or \
                                        (self.input_type=='EEG_IMG' and (self.dataset_mode_EEG_IMG!='load')) or\
                                        (self.input_type=='ALL'     and (self.dataset_mode_EEG_WAVE!='load'))
        self.load_eeg_wave          = (self.input_type=='EEG_WAVE'and (self.dataset_mode_EEG_WAVE=='load')) or\
                                        (self.input_type=='EEG_IMG' and (self.dataset_mode_EEG_WAVE=='load')) or\
                                        (self.input_type=='ALL'     and (self.dataset_mode_EEG_WAVE=='load'))
        self.calc_eeg_img           = (self.input_type=='EEG_IMG' and (self.dataset_mode_EEG_IMG!='load')) or\
                                        (self.input_type=='ALL'     and (self.dataset_mode_EEG_IMG!='load'))
        self.load_eeg_img           = (self.input_type=='EEG_IMG' and (self.dataset_mode_EEG_IMG=='load')) or\
                                        (self.input_type=='ALL'     and (self.dataset_mode_EEG_IMG=='load'))
        print(f'calc_spec:{self.calc_spec},calc_eeg_wave:{self.calc_eeg_wave},load_eeg_wave:{self.load_eeg_wave},calc_eeg_img:{self.calc_eeg_img},load_eeg_img:{self.load_eeg_img}')

        #==df==#
        if self.phase != 'test':
            col_labels                   = config['dataset']['col_labels']
            self.df                      = df
            self.label_id                = df['label_id'].values
            self.patient_id              = df['patient_id'].values
            self.eeg_id                  = df['eeg_id'].values
            self.spectrogram_id          = df['spectrogram_id'].values
            self.votes                   = df[col_labels].values
            self.eeg_offset_sec                         = df['eeg_label_offset_seconds'].values #EEGの開始からこのサブサンプルまでの時間。専門家が見た50[ms]の内、EEGの開始位置
            self.spectrogram_offset_sec                 = df['spectrogram_label_offset_seconds'].values
            self.eeg_sub_id                             = df['eeg_sub_id'].values
            self.spectrogram_sub_id                     = df['spectrogram_sub_id'].values
        else:
            self.df                     = df
            self.patient_id             = df['patient_id'].values
            self.eeg_id                 = df['eeg_id'].values
            self.spectrogram_id         = df['spectrogram_id'].values

        #==data==#
        self.eegs_data_dict             = eegs_data_dict
        self.spectrograms_data_dict     = spectrograms_data_dict 
        self.dict_col_index             = dict_col_index

        #==label smoothing==#
        self.label_smoothing_ver        = config['calc_mode']['label_smoothing_ver'] #"ver_2"
        self.label_smoothing_n_evaluator= config['calc_mode']['label_smoothing_n_evaluator']
        self.label_smoothing_k          = config['calc_mode']['label_smoothing_k']
        self.label_smoothing_epsilon    = config['calc_mode']['label_smoothing_epsilon']

        #==Setting SPEC==#
        if self.calc_spec:
            self.sr_spec                = config['dataset']['SPEC']['sr_spec']
            self.img_size_spec          = config['dataset']['SPEC']['img_size_spec']
            self.in_chans_spec_2        = config['dataset']['SPEC']['in_chans_spec_2']
            #resize
            if self.img_size_spec[0]  <0:
                self.resize_spec        = None
            else:
                self.resize_spec        = transforms.Resize((self.img_size_spec[0] ,self.img_size_spec[1]))
            #augmentation
            self.random_channel_spec    = config['dataset']['SPEC']['augmentation']['random_channel']
            self.random_ranges_spec     = config['dataset']['SPEC']['augmentation']['random_channel_range']
            self.stop_aug_epoch_spec    = config['dataset']['SPEC']['augmentation']['stop_aug_epoch']
        #==Setting EEG_WAVE==#
        if self.calc_eeg_wave: 
            self.wave_select            = config['dataset']['EEG']['wave_select'] #'delta' or 'raw'
            self.wave_select_wave       = config['dataset']['EEG']['wave_select_wave'] #'delta' or 'raw'
            self.wave_select_img        = config['dataset']['EEG']['wave_select_img'] #'delta' or 'raw'
            self.in_chans_eeg_wave      = config['dataset']['EEG']['in_chans_eeg_wave']
            self.in_chans_eeg_img       = config['dataset']['EEG']['in_chans_eeg_img']
            self.sr_eeg_org             = config['dataset']['EEG']['sr_eeg_org']
            self.sr_eeg_resample        = config['dataset']['EEG']['sr_eeg_resample']
            #filter
            self.calc_bandpass_filter   = config['dataset']['EEG']['bandpass_filter']['calc_bandpass_filter']
            self.calc_bandpass_f_up     = config['dataset']['EEG']['bandpass_filter']['f_up']
            self.calc_bandpass_f_low    = config['dataset']['EEG']['bandpass_filter']['f_low']
            #augmentation
            self.random_channel_eeg_wave= config['dataset']['EEG']['augmentation']['random_channel']
            self.random_ranges_eeg_wave = config['dataset']['EEG']['augmentation']['random_channel_range']
            self.wave_inverse           = config['dataset']['EEG']['augmentation']['wave_inverse']

            self.stop_aug_epoch_eeg_wave= config['dataset']['EEG']['augmentation']['stop_aug_epoch']
            #時間抽出2
            self.calc_select_time2_wave = config['dataset']['EEG']['select_time2_wave']['calc_select_time2_wave']
            self.use_time_ave_wave      = config['dataset']['EEG']['select_time2_wave']['use_time_ave_wave']
        #==Setting EEG_IMG==#
        if (self.calc_eeg_img) or (self.load_eeg_img):
            #==EEG_IMG==#
            self.in_chans_eeg_img2      = config['dataset']['EEG']['in_chans_eeg_img2']
            #calc_mode
            self.calc_img1_stft         = config['dataset']['EEG_IMG']['IMG1_stft']['calc_img1_stft']
            #select
            self.img_select             = config['dataset']['EEG_IMG']['img_select']
            self.add_eeg_img_feat       = config['dataset']['EEG_IMG']['add_eeg_img_feat']
            #resize
            self.img_size_eeg           = config['dataset']['EEG_IMG']['img_size']
            if self.img_size_eeg[0]  <0:
                self.resize_eeg         = None
            else:
                self.resize_eeg         = transforms.Resize((self.img_size_eeg[0] ,self.img_size_eeg[1]))
            #augmentation
            self.random_channel_eeg_img     = config['dataset']['EEG_IMG']['augmentation']['random_channel']
            self.random_ranges_eeg_img      = config['dataset']['EEG_IMG']['augmentation']['random_channel_range']
            self.stop_aug_epoch_eeg_img     = config['dataset']['EEG_IMG']['augmentation']['stop_aug_epoch']
            #時間抽出2
            self.calc_select_time2_img      = config['dataset']['EEG_IMG']['select_time2_img']['calc_select_time2_img']
            self.use_time_ave_img           = config['dataset']['EEG_IMG']['select_time2_img']['use_time_ave_img']

        #==Setting ALL==#
        if self.input_type=='ALL':
            self.stop_aug_epoch_spec        = config['dataset']['ALL']['augmentation']['stop_aug_epoch']
            self.stop_aug_epoch_eeg_wave    = config['dataset']['ALL']['augmentation']['stop_aug_epoch']
            self.stop_aug_epoch_eeg_img     = config['dataset']['ALL']['augmentation']['stop_aug_epoch']
            self.aug_prob_spec              = config['dataset']['ALL']['augmentation']['prob']
            self.aug_prob_eeg_wave          = config['dataset']['ALL']['augmentation']['prob']
            self.aug_prob_eeg_img           = config['dataset']['ALL']['augmentation']['prob']

        #==augmentation==#
        if self.phase == 'train':
            self.aug_prob_spec          = config['dataset']['SPEC']['augmentation']['prob']
            self.aug_prob_eeg_wave      = config['dataset']['EEG']['augmentation']['prob']
            self.aug_prob_eeg_img       = config['dataset']['EEG_IMG']['augmentation']['prob']
            self.img_trans_spec         = get_spectrogram_transforms(config['dataset']['SPEC']['augmentation'],self.aug_prob_spec,self.phase)
            self.wave_trans_eeg         = get_wave_transforms(config['dataset']['EEG']['augmentation'],self.aug_prob_eeg_wave,self.phase)
            self.img_trans_eeg          = get_spectrogram_transforms(config['dataset']['EEG_IMG']['augmentation'],self.aug_prob_eeg_img,self.phase)
        else:
            self.aug_prob_spec          = config['dataset']['SPEC']['augmentation']['prob']
            self.aug_prob_eeg_wave      = config['dataset']['EEG']['augmentation']['prob']
            self.aug_prob_eeg_img       = config['dataset']['EEG_IMG']['augmentation']['prob']
            self.img_trans_spec         = None
            self.wave_trans_eeg         = None
            self.img_trans_eeg          = None
        
        #==data selectsetting==#
        self.groups                     = df.groupby(['patient_id', 'spectrogram_id']).groups
        self.groups_eeg_id              = df.groupby(['patient_id', 'eeg_id']).groups
        self.selected_indices           = []
        self.selection_counts           = {}  
        self.img_flatten_eeg = False

    def __len__(self):
        if self.phase != 'test':
            return len(self.selected_indices)
        else:
            return len(self.df)

    #============dataset============#
    def __getitem__(self, idx):
        #===data_select===
        if self.phase != 'test':
            data_idx                    = self.selected_indices[idx]
            #pos
            label_id                    = self.label_id[data_idx]
            patient_id                  = self.patient_id[data_idx]
            eeg_id                      = self.eeg_id[data_idx]
            spectrogram_id              = self.spectrogram_id[data_idx]
            eeg_offset_sec              = self.eeg_offset_sec[data_idx]
            spectrogram_offset_sec      = self.spectrogram_offset_sec[data_idx]
            #label
            votes                       = self.votes[data_idx]
            num_votes                   = np.sum(votes)
            label                       = votes/num_votes
            #===Label smoothing===#
            if self.phase == "train":
                label                   = self.label_smoothing(label, num_votes)
                
        else:#test
            data_idx                    = idx
            #pos
            label_id                    = []
            patient_id                  = self.patient_id[data_idx]
            eeg_id                      = self.eeg_id[data_idx]
            spectrogram_id              = self.spectrogram_id[data_idx]
            eeg_offset_sec              = 0
            spectrogram_offset_sec      = 0
            #label
            label                       = []
            
        #===INIT===#
        sub_spectrogram_4ch             = []
        sub_eeg_data_wave               = []
        sub_eeg_data_img                = []    
        #===SPEC===#
        if self.calc_spec:
            spectrogram_data            = self.spectrograms_data_dict[spectrogram_id]
            #pick_up
            if self.phase != 'test':
                sub_start_spec          = int(spectrogram_offset_sec*self.sr_spec)
            else:
                sub_start_spec          = int(spectrogram_offset_sec*self.sr_spec)
            sub_end_spec                = sub_start_spec+int(600*self.sr_spec)
            sub_spectrogram_data        = spectrogram_data[sub_start_spec:sub_end_spec].T#10min間のspec生データ (time,freq)=(401, 300)

            sub_spectrogram_LL          = sub_spectrogram_data[self.dict_col_index['ind_spec_LL'],:]
            sub_spectrogram_RL          = sub_spectrogram_data[self.dict_col_index['ind_spec_RL'],:]
            sub_spectrogram_RP          = sub_spectrogram_data[self.dict_col_index['ind_spec_RP'],:]
            sub_spectrogram_LP          = sub_spectrogram_data[self.dict_col_index['ind_spec_LP'],:]
            
            #===stack===#
            sub_spectrogram_4ch         = np.stack([sub_spectrogram_LL,
                                                    sub_spectrogram_RL,
                                                    sub_spectrogram_RP,
                                                    sub_spectrogram_LP],axis=0) #(ch,freq,time)=(4,100,300) 0～20Hz、10min
            sub_spectrogram_4ch         = np.nan_to_num(sub_spectrogram_4ch, nan=0.0)
            sub_spectrogram_4ch         = np.clip(sub_spectrogram_4ch,np.exp(-4),np.exp(8)) 

            #===LOG TRANSFORM SPECTROGRAM===#
            sub_spectrogram_4ch         = np.log(sub_spectrogram_4ch)

            #===norm_img===#
            calc_norm_spec              = False
            if calc_norm_spec:#各waveファイルごとに規格化
                ep                      = 1e-6
                max                     = np.nanmean(sub_spectrogram_4ch.flatten())
                std                     = np.nanstd(sub_spectrogram_4ch.flatten())
                sub_spectrogram_4ch     = (sub_spectrogram_4ch-max)/(std+ep)
                sub_spectrogram_4ch     = np.nan_to_num(sub_spectrogram_4ch, nan=0.0)
            else:#全データ一律規格化
                ep                      = 1e-6
                max                     = 8
                min                     = -4
                sub_spectrogram_4ch     = np.clip(sub_spectrogram_4ch,1*min,1*max) 
                sub_spectrogram_4ch     = (sub_spectrogram_4ch-0.5*(max+min+ep))/(max-min+ep) #-1~1に規格化
                sub_spectrogram_4ch     = np.nan_to_num(sub_spectrogram_4ch, nan=0.0)
            #==========================================================================================================#
            #=torch=#
            sub_spectrogram_4ch         = torch.tensor(sub_spectrogram_4ch)

            #=augmentation(img)=#
            if self.epoch < self.stop_aug_epoch_spec:
                sub_spectrogram_4ch     = augmentation_spec(sub_spectrogram_4ch,self.phase,
                                                                self.img_trans_spec,
                                                                self.random_channel_spec,
                                                                self.random_ranges_spec,
                                                                self.resize_spec,
                                                                self.aug_prob_spec)
        #===EEG_WAVE===#
        if self.calc_eeg_wave:
            #==make_feature2_1 Preprocess==#
            eeg_data                            = self.eegs_data_dict[eeg_id]
            #===pick_up===#
            if self.phase != 'test':
                sub_start_eeg                   = int(eeg_offset_sec*self.sr_eeg_org)
            else:
                sub_start_eeg                   = int(eeg_offset_sec*self.sr_eeg_org)
            #
            sub_end_eeg                         = sub_start_eeg+int(50*self.sr_eeg_org)
            sub_eeg_data                        = eeg_data[sub_start_eeg:sub_end_eeg]#50秒間のeeg生データ (time,feat)=(10000, 20) 
            
            #===select_data===#
            sub_eeg_data_wave                   = select_eeg_data(sub_eeg_data,self.wave_select,self.dict_col_index)
            
            #===norm===#  
            sub_eeg_data_wave                   = np.clip(sub_eeg_data_wave, -1024, 1024)
            sub_eeg_data_wave                   = np.nan_to_num(sub_eeg_data_wave, nan=0) / 32.0

            #===raw_wave===#
            sub_eeg_data_wave_raw               = np.nan_to_num(sub_eeg_data_wave)

            #===select_ch===#
            if self.wave_select!=self.wave_select_wave:#24ch計算して16ch使用
                sub_eeg_data_wave               = sub_eeg_data_wave[:self.in_chans_eeg_wave]

            if self.wave_select!=self.wave_select_img:#24ch計算して16ch使用
                sub_eeg_data_wave_raw           = sub_eeg_data_wave_raw[:self.in_chans_eeg_img]

            #===時間抽出2===#
            if self.calc_select_time2_wave:
                length                          = 2000
                if self.phase=='train':
                    start_ind                   = random.randint(1000, 7000)
                    end_ind                     = start_ind+length
                    sub_eeg_data_wave           = sub_eeg_data_wave[:,start_ind:end_ind]
                elif self.phase=='valid':
                    if self.use_time_ave_wave==False:#中心2000sample
                        start_ind               = 4000
                        end_ind                 = start_ind+length
                        sub_eeg_data_wave       = sub_eeg_data_wave[:,start_ind:end_ind]
                    else:#5分割weight_ave
                        pass

            #===augmentation===#
            if self.dataset_mode_EEG_WAVE!='save':#saveする時はaugmentationしない
                if self.epoch < self.stop_aug_epoch_eeg_wave:
                    sub_eeg_data_wave           = augmentation_eeg_wave(sub_eeg_data_wave,self.phase,
                                                                        self.wave_trans_eeg,self.sr_eeg_org,
                                                                        self.wave_inverse,
                                                                        self.random_channel_eeg_wave,
                                                                        self.random_ranges_eeg_wave,
                                                                        self.aug_prob_eeg_wave)
            else:
                pass
        #===Load EEG_WAVE===#
        if self.load_eeg_wave:
            sub_eeg_data_wave                   = self.dict_EEG_WAVE[eeg_id]
            #===augmentation===#
            if self.epoch < self.stop_aug_epoch_eeg_wave:
                sub_eeg_data_wave               = augmentation_eeg_wave(sub_eeg_data_wave,self.phase,
                                                                        self.wave_trans_eeg,self.sr_eeg_org,
                                                                        self.wave_inverse,
                                                                        self.random_channel_eeg_wave,
                                                                        self.random_ranges_eeg_wave,
                                                                        self.aug_prob_eeg_wave)
        #===EEG_IMG===#
        if self.calc_eeg_img:
            #===fet_feat===#
            sub_eeg_data_img_1                  = get_eeg_img_feat(sub_eeg_data_wave_raw,self.sr_eeg_resample,self.config,
                                                                self.calc_img1_stft)
            #===まとめ===#
            if self.img_select=='img1':
                sub_eeg_data_img_tmps           = [sub_eeg_data_img_1]    
            
            sub_eeg_data_img                    = []
            for idz in range(len(sub_eeg_data_img_tmps)):
                #===torch===#
                sub_eeg_data_img_tmp            = sub_eeg_data_img_tmps[idz]#(16,100,300)

                #===torch===#
                sub_eeg_data_img_tmp            = torch.tensor(sub_eeg_data_img_tmp)

                #===saveしない場合===#
                if self.dataset_mode_EEG_IMG!='save':#saveしない時はaugmentationする
                    #===augmentation===#
                    if self.epoch < self.stop_aug_epoch_eeg_img:
                        sub_eeg_data_img_tmp    = augmentation_eeg_img(sub_eeg_data_img_tmp,self.phase,
                                                                        self.img_trans_eeg,
                                                                        self.random_channel_eeg_img,
                                                                        self.random_ranges_eeg_img,
                                                                        self.resize_eeg,
                                                                        self.aug_prob_eeg_img,)
                    #===時間抽出2===#
                    if self.calc_select_time2_img:
                        total_length                    = sub_eeg_data_img_tmp.shape[-1]
                        length                          = total_length//5
                        if self.phase=='train':
                            start_ind                   = random.randint(length//2, total_length-length//2)
                            end_ind                     = start_ind+length
                            sub_eeg_data_img_tmp        = sub_eeg_data_img_tmp[:,start_ind:end_ind]
                        elif self.phase=='valid':
                            if self.use_time_ave_img==False:#中心2000sample
                                start_ind               = 2*total_length//5
                                end_ind                 = start_ind+length
                                sub_eeg_data_img_tmp    = sub_eeg_data_img_tmp[:,start_ind:end_ind]
                            else:#5分割weight_ave
                                pass
                    #===concat===#
                    if idz==0:
                        sub_eeg_data_img        = sub_eeg_data_img_tmp.unsqueeze(1)#(16,1,100,300)
                    else:
                        sub_eeg_data_img        = torch.cat([sub_eeg_data_img,sub_eeg_data_img_tmp.unsqueeze(1)],dim=1)#(16,x,100,300)
                #===saveする場合===#
                else:#saveする時listで保存しておく
                    sub_eeg_data_img.append(sub_eeg_data_img_tmp)

        #===Load EEG_IMG===#
        if self.load_eeg_img:   
            sub_eeg_data_img_tmps               = self.dict_EEG_IMG[eeg_id]#list配列
            for idz in range(len(sub_eeg_data_img_tmps)):
                sub_eeg_data_img_tmp            = sub_eeg_data_img_tmps[idz]#(16,100,300)
                #===augmentation===#
                if self.epoch < self.stop_aug_epoch_eeg_img:
                    sub_eeg_data_img_tmp        = augmentation_eeg_img(sub_eeg_data_img_tmp,self.phase,
                                                                        self.img_trans_eeg,
                                                                        self.random_channel_eeg_img,
                                                                        self.random_ranges_eeg_img,
                                                                        self.resize_eeg,
                                                                        self.aug_prob_eeg_img,)
                #===時間抽出2===#
                #print(sub_eeg_data_img_tmp.shape)
                if self.calc_select_time2_img:
                    total_length                    = sub_eeg_data_img_tmp.shape[-1]
                    length                          = total_length//5
                    if self.phase=='train':
                        start_ind                   = random.randint(length//2, total_length-3*length//2)
                        end_ind                     = start_ind+length
                        sub_eeg_data_img_tmp        = sub_eeg_data_img_tmp[:,:,start_ind:end_ind]
                    elif self.phase=='valid':
                        if self.use_time_ave_img==False:#中心2000sample
                            start_ind               = 2*total_length//5
                            end_ind                 = start_ind+length
                            sub_eeg_data_img_tmp    = sub_eeg_data_img_tmp[:,:,start_ind:end_ind]
                        else:#
                            pass
                #===concat===#
                if idz==0:
                    sub_eeg_data_img            = sub_eeg_data_img_tmp.unsqueeze(1)#(16,1,100,300)
                else:
                    sub_eeg_data_img            = torch.cat([sub_eeg_data_img,sub_eeg_data_img_tmp.unsqueeze(1)],dim=1)#(16,x,100,300)        
        #===RETURN===#
        return {
            "label_id"              : label_id,
            "patient_id"            : patient_id,
            "eeg_id"                : eeg_id,
            "spectrogram_id"        : spectrogram_id,
            'sub_spectrogram_4ch'  : sub_spectrogram_4ch,
            'sub_eeg_data_wave'    : torch.tensor(sub_eeg_data_wave),
            'sub_eeg_data_img'     : sub_eeg_data_img,
            "label"                : torch.FloatTensor(label),  # (pred_length, num_classes)
        }
    
    #====data選択====#
    def update_for_epoch(self,epoch):
        self.epoch                  = epoch
        #seed
        if self.phase == 'train':
            random.seed(epoch)#trainはepochごとに選択変更
        else:
            random.seed(self.seed+5)#validはepochによらず選択固定
        self.selected_indices       = []
        if self.select_data         == 'use_alldata':#全部使う
            self.selected_indices   = list(range(len(self.df)))
        elif self.select_data       == 'patient_id_eeg_id':
            for group_key, indices in self.groups_eeg_id.items():
                self.selected_indices.append(random.choice(list(indices)))

    #====label_smoothing====#
    def do_custom_label_smoothing(self, label, epsilon):
        # ピークが複数ある場合、全てのピークを落としてそれ以外を持ち上げる
        max_value       = np.max(label)
        idx_max         = np.where(label == max_value)[0]
        is_max          = np.identity(6)[idx_max].sum(axis=0).astype("bool")
        n_idx_max       = is_max.sum()
        new_epsilon     = epsilon / n_idx_max
        label[is_max]   -= new_epsilon
        label[~is_max]  += epsilon / (6 - n_idx_max)
        return label

    def label_smoothing(self, label, n_evaluator):
        if self.label_smoothing_ver == "ver_1":
            epsilon     = 1 / (self.label_smoothing_k + np.sqrt(n_evaluator))
            label       = self.do_custom_label_smoothing(label, epsilon)
        elif self.label_smoothing_ver == "ver_2":
            if n_evaluator <= self.label_smoothing_n_evaluator:
                label   = self.do_custom_label_smoothing(
                    label, self.label_smoothing_epsilon
                )
        else:
            pass
        return label