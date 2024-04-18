# ===============
# Import
# ===============
from torch.utils.data import Dataset


# ==============================
# Train Dataset
# ==============================
class HMS_Dataset_PP(Dataset):
    def __init__(self,df,col_feat,col_label,phase='train'):
        #==other==#
        self.df                      = df
        self.feat                    = df[col_feat].values
        self.patient_id              = df['patient_id'].values
        self.eeg_id                  = df['eeg_id'].values
        self.spectrogram_id          = df['spectrogram_id'].values
        self.phase                   = phase
        if phase != 'test':
            self.label                = df[col_label].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feat                       = self.feat[idx]
        patient_id                 = self.patient_id[idx]
        eeg_id                     = self.eeg_id[idx]
        spectrogram_id             = self.spectrogram_id[idx]

        if self.phase != 'test':
            label                  = self.label[idx]
            return {
                'patient_id'         : patient_id,
                'eeg_id'             : eeg_id,
                'spectrogram_id'     : spectrogram_id,
                'feat'               : feat,
                'label'              : label,
                } 
        else:
            return {
                'patient_id'         : patient_id,
                'eeg_id'             : eeg_id,
                'spectrogram_id'     : spectrogram_id,
                'feat'               : feat,
                } 
    