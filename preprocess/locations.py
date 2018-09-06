class data_locations:
    def __init__(self):

        # Locations of digitized sentences of training/valdiation data
        self.train_context_path ='train_data/train_context.npy'# 'train_data/context.train'# 
        self.train_context_len_path = 'train_data/train_context_len.npy'
        self.train_context_speaker_path = 'train_data/train_context_speaker.npy'# 'train_data/context_speaker.train'#

        self.train_target_path = 'train_data/train_target.npy'
        
        self.train_options_path = 'train_data/train_options.npy'# 'train_data/options.train'#
        self.train_options_len_path = 'train_data/train_options_len.npy'
        self.train_options_speaker_path = 'train_data/train_options_speaker.npy'# 'train_data/options_speaker.train'#

        self.train_profile_path = 'train_data/train_profile.npy'


        self.valid_context_path = 'valid_data/valid_context.npy'# 'valid_data/context.valid'#
        self.valid_context_len_path = 'valid_data/valid_context_len.npy'
        self.valid_context_speaker_path = 'valid_data/valid_context_speaker.npy'# 'valid_data/context_speaker.valid'#

        self.valid_target_path = 'valid_data/valid_target.npy'
        
        self.valid_options_path = 'valid_data/valid_options.npy'# 'valid_data/options.valid'#
        self.valid_options_len_path = 'valid_data/valid_options_len.npy'
        self.valid_options_speaker_path = 'valid_data/valid_options_speaker.npy'# 'valid_data/options_speaker.valid'#

        self.valid_profile_path = 'valid_data/valid_profile.npy'

        


def create_data_locations():
    return data_locations()
