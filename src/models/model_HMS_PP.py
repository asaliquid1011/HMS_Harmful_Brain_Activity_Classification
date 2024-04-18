# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import sys
import torch
import torch.nn as nn
sys.path.append("..")

# ==============================
# PP
# ==============================
class HMSModel_PP(torch.nn.Module):
    def __init__(self,col_feat,config):
        super(HMSModel_PP, self).__init__()

        in_chans                = len(col_feat)
        n_classes               = config['model']['PP']['n_classes']
        hidden_mlp              = config['model']['PP']['hidden_mlp']
        num_layers              = config['model']['PP']['num_layers']
        if num_layers ==1:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,n_classes)
                                            )
            
        elif num_layers ==2:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,hidden_mlp),
                                            nn.Linear(hidden_mlp, n_classes)
                                            )
        elif num_layers ==3:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,hidden_mlp),
                                            nn.Linear(hidden_mlp,hidden_mlp),
                                            nn.Linear(hidden_mlp, n_classes)
                                            )
        elif num_layers ==4:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,hidden_mlp),
                                            nn.Linear(hidden_mlp,hidden_mlp),
                                            nn.Linear(hidden_mlp,hidden_mlp),
                                            nn.Linear(hidden_mlp, n_classes)
                                            )

                
    def forward(self,feat):
        out                     =  self.mlp(feat) 
        return (out.softmax(dim=1), out)
    
# ==============================
# Stacking
# ==============================
class HMSModel_Stacking(torch.nn.Module):
    def __init__(self,col_feat,config):
        super(HMSModel_Stacking, self).__init__()

        in_chans                = len(col_feat)
        n_classes               = config['model']['Stacking']['n_classes']
        hidden_mlp              = config['model']['Stacking']['hidden_mlp']
        num_layers              = config['model']['Stacking']['num_layers']
        if num_layers ==1:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,n_classes)
                                            )
            
        elif num_layers ==2:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,hidden_mlp),
                                            nn.Linear(hidden_mlp, n_classes)
                                            )
        elif num_layers ==3:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,hidden_mlp),
                                            nn.Linear(hidden_mlp,hidden_mlp),
                                            nn.Linear(hidden_mlp, n_classes)
                                            )
        elif num_layers ==4:
            self.mlp                = nn.Sequential(
                                            nn.Linear(in_chans,hidden_mlp),
                                            nn.Linear(hidden_mlp,hidden_mlp),
                                            nn.Linear(hidden_mlp,hidden_mlp),
                                            nn.Linear(hidden_mlp, n_classes)
                                            )
                
    def forward(self,feat):
        out                     =  self.mlp(feat) 
        return (out.softmax(dim=1), out)
    
