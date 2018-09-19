class data_locations:
    def __init__(self):

        self.train_context_path ='train_data/train_ubuntu_uKB_test/train_context.npy'# 'train_data/context.train'# 
        self.train_context_speaker_path = 'train_data/train_ubuntu_uKB_test/train_context_speaker.npy'# 'train_data/context_speaker.train'#
        self.train_context_mask_path = 'train_data/train_ubuntu_uKB_test/train_context_mask.npy'
        self.train_context_len_path = 'train_data/train_ubuntu_uKB_test/train_context_len.npy'

        self.train_target_path = 'train_data/train_ubuntu/train_target.npy'
        
        self.train_options_path = 'train_data/train_ubuntu_uKB_test/train_options.npy'# 'train_data/options.train'#
        self.train_options_len_path = 'train_data/train_ubuntu_uKB_test/train_options_len.npy'

        self.train_profile_path = 'train_data/train_profile.npy'


        self.valid_context_path = 'valid_data/valid_ubuntu_uKB_test/valid_context.npy'# 'valid_data/context.valid'#
        self.valid_context_speaker_path = 'valid_data/valid_ubuntu_uKB_test/valid_context_speaker.npy'# 'valid_data/context_speaker.valid'#
        self.valid_context_mask_path = 'valid_data/valid_ubuntu_uKB_test/valid_context_mask.npy'# 'valid_data/context_speaker.valid'#
        self.valid_context_len_path = 'valid_data/valid_ubuntu_uKB_test/valid_context_len.npy'

        self.valid_target_path = 'valid_data/valid_target.npy'
        
        self.valid_options_path = 'valid_data/valid_ubuntu_uKB_test/valid_options.npy'# 'valid_data/options.valid'#
        self.valid_options_len_path = 'valid_data/valid_ubuntu_uKB_test/valid_options_len.npy'

        self.valid_profile_path = 'valid_data/valid_profile.npy'



        '''
        # Locations of digitized sentences of training/valdiation data
        self.train_zp_context_path = 'train_data/train_zp_context.npy'
        self.train_context_zp_path = 'train_data/train_context_zp.npy'
        self.train_zp_context_speaker_path = 'train_data/train_zp_context_speaker.npy'
        self.train_context_len_path = 'train_data/train_context_len.npy'

        self.train_target_path = 'train_data/train_ubuntu/train_target.npy'
        
        self.train_zp_options_path = 'train_data/train_zp_options.npy'
        self.train_options_zp_path = 'train_data/train_options_zp.npy'
        self.train_options_len_path = 'train_data/train_ubuntu_uKB/train_options_len.npy'

        self.train_profile_path = 'train_data/train_ubuntu/train_profile.npy'


        self.valid_zp_context_path = 'valid_data/valid_zp_context.npy'
        self.valid_context_zp_path = 'valid_data/valid_context_zp.npy'
        self.valid_zp_context_speaker_path = 'valid_data/valid_zp_context_speaker.npy'
        self.valid_context_len_path = 'valid_data/valid_context_len.npy'

        self.valid_target_path = 'valid_data/valid_target.npy'

        self.valid_zp_options_path = 'valid_data/valid_zp_options.npy'
        self.valid_options_len_path = 'valid_data/valid_options_len.npy'

        self.valid_profile_path = 'valid_data/valid_ubuntu/valid_profile.npy'


        # Locations of digitized sentences of training/valdiation data
        '''
        


def create_data_locations():
    return data_locations()
