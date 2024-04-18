# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import sys
import torch
import torch.nn as nn
sys.path.append("..")
from models.model_HMS_CNN     import HMSModel_CNN
from models.model_HMS_WAVE_CNN import HMSModel_WAVE_CNN

# ==============================
# multi
# ==============================
class HMSModel_MULTI(torch.nn.Module):
    def __init__(self,config,id_fold):
        super(HMSModel_MULTI, self).__init__()
        self.config                     = config
        self.calc_spec                  = config['model']['MULTI']['calc_spec']
        self.calc_eeg_wave              = config['model']['MULTI']['calc_eeg_wave']
        self.calc_eeg_img               = config['model']['MULTI']['calc_eeg_img']
        mode_load_multi_model           = config['model']['MULTI']['mode_load_multi_model']
        output_dir                      = config['path']['output_dir']
        exp_id_spec                     = config['model']['MULTI']['load_exp_id']['exp_id_spec']
        exp_id_eeg_wave                 = config['model']['MULTI']['load_exp_id']['exp_id_eeg_wave']
        exp_id_eeg_img                  = config['model']['MULTI']['load_exp_id']['exp_id_eeg_img']
        self.output_type                = config['model']['MULTI']['output_type']
        self.n_classes                  = config['model']['MULTI']['output_chans']
        self.multi_output               = config['model']['MULTI']['multi_output']
        path_model_spec                 = f'{output_dir}/{exp_id_spec}/fold_{id_fold}/best.pt'
        path_model_eeg_wave             = f'{output_dir}/{exp_id_eeg_wave}/fold_{id_fold}/best.pt'
        path_model_eeg_img              = f'{output_dir}/{exp_id_eeg_img}/fold_{id_fold}/best.pt'
        if mode_load_multi_model:
            if self.calc_spec:
                self.model_spec                 = torch.load(path_model_spec)
                self.model_spec.output_type     = self.output_type 
                # self.model_spec                 = self.model_spec.train()
            if self.calc_eeg_wave:
                self.model_eeg_wave             = torch.load(path_model_eeg_wave)
                self.model_eeg_wave.output_type = self.output_type 
                # self.model_eeg_wave             = self.model_eeg_wave.train()
            if self.calc_eeg_img:
                self.model_eeg_img              = torch.load(path_model_eeg_img)
                self.model_eeg_img.output_type  = self.output_type 
                # self.model_eeg_img              = self.model_eeg_img.train()
        else:
            if self.calc_spec:
                self.model_spec                 = HMSModel_CNN(config,input_type='SPEC')
                self.model_spec.output_type     = self.output_type 
            if self.calc_eeg_wave:
                self.model_eeg_wave             = HMSModel_WAVE_CNN(config)
                self.model_eeg_wave.output_type = self.output_type 
            if self.calc_eeg_img:
                self.model_eeg_img              = HMSModel_CNN(config,input_type='EEG_IMG')
                self.model_eeg_img.output_type  = self.output_type 

        # 'logits_output':
        self.head                   = config['model']['MULTI']['logits_output']['head']
        #=mlp head=#
        if self.head=='mlp':
            self.mode_add_feat      = config['model']['MULTI']['logits_output']['mlp_head']['mode_add_feat']
            in_chans_head           = config['model']['MULTI']['logits_output']['mlp_head']['in_chans_head'] #24+256+96
            hidden_mlphead          = config['model']['MULTI']['logits_output']['mlp_head']['hidden_mlphead']
            dropout_mlphead         = config['model']['MULTI']['logits_output']['mlp_head']['dropout_mlphead']
            self.head_mlp           = nn.Sequential(
                                                nn.Linear(in_chans_head,hidden_mlphead),
                                                nn.LayerNorm(hidden_mlphead),
                                                nn.ReLU(),
                                                nn.Dropout(dropout_mlphead),
                                                nn.Linear(hidden_mlphead, self.n_classes)
                                                )


    def forward(self,spec,eeg_wave,eeg_img):
        if self.calc_spec:
            _,out_spec                  = self.model_spec(spec)
        if self.calc_eeg_wave:
            _,out_eeg_wave              = self.model_eeg_wave(eeg_wave)
        if self.calc_eeg_img:
            _,out_eeg_img               = self.model_eeg_img(eeg_img)
        
        #=mlp head=#
        if self.head=='mlp':
            out = []
            if self.calc_spec:
                out.append(out_spec)
            if self.calc_eeg_wave:
                out.append(out_eeg_wave)
            if self.calc_eeg_img:
                out.append(out_eeg_img)
            out                         = torch.cat(out,dim=1)
            out                         = self.head_mlp(out)

            if self.multi_output:
                out_spec                = self.head_mlp_spec(out_spec)
                out_eeg_wave            = self.head_mlp_wave(out_eeg_wave)
                out_eeg_img             = self.head_mlp_img(out_eeg_img)
                return (out.softmax(dim=1), out ,out_spec, out_eeg_wave, out_eeg_img)
            else:
                return (out.softmax(dim=1), out)


