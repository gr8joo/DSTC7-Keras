class data_locations:
    def __init__(self):

        # Locations of digitized sentences of training/valdiation data
        self.train_context_path = 'train_data/train_context.npy'
        self.train_context_len_path = 'train_data/train_context_len.npy'
        self.train_target_path = 'train_data/train_target.npy'
        self.train_options_path = 'train_data/train_options.npy'
        self.train_options_len_path = 'train_data/train_options_len.npy'

        self.valid_context_path = 'valid_data/valid_context.npy'
        self.valid_context_len_path = 'valid_data/valid_context_len.npy'
        self.valid_target_path = 'valid_data/valid_target.npy'
        self.valid_options_path = 'valid_data/valid_options.npy'
        self.valid_options_len_path = 'valid_data/valid_options_len.npy'

        
        # Locations of 1up sentences of training/valdiation data
        self.train_context_1up_path = 'train_data/train_advising_1up/train_context_eou.npy'
        self.train_target_1up_path = 'train_data/train_advising/train_target.npy'
        self.train_options_1up_path = 'train_data/train_advising_1up/train_options_eou.npy'

        self.valid_context_1up_path = 'valid_data/valid_advising_1up/valid_context_eou.npy'
        self.valid_target_1up_path = 'valid_data/valid_advising/valid_target.npy'
        self.valid_options_1up_path = 'valid_data/valid_advising_1up/valid_options_eou.npy'

        
        # Locations of no eot sentences of training/valdiation data
        self.train_context_no_eot_path = 'train_data/train_advising_no_eot/train_context_no_eot.npy'
        self.train_target_no_eot_path = 'train_data/train_advising/train_target.npy'
        # share        
        self.train_options_no_eot_path = 'train_data/train_advising_1up/train_options_eou.npy'# 'train_data/train_advising_no_eot/train_options_no_eot.npy'

        self.valid_context_no_eot_path = 'valid_data/valid_advising_no_eot/valid_context_no_eot.npy'
        self.valid_target_no_eot_path = 'valid_data/valid_advising/valid_target.npy'
        # share
        self.valid_options_no_eot_path = 'valid_data/valid_advising_1up/valid_options_eou.npy'# 'valid_data/valid_advising_no_eot/valid_options_no_eot.npy'

        
        # Locations of no eou sentences of training/valdiation data
        self.train_context_no_eou_path = 'train_data/train_advising_no_eou/train_context_no_eou.npy'
        self.train_target_no_eou_path = 'train_data/train_advising/train_target.npy'
        self.train_options_no_eou_path = 'train_data/train_advising_no_eou/train_options_no_eou.npy'

        self.valid_context_no_eou_path = 'valid_data/valid_advising_no_eou/valid_context_no_eou.npy'
        self.valid_target_no_eou_path = 'valid_data/valid_advising/valid_target.npy'
        self.valid_options_no_eou_path = 'valid_data/valid_advising_no_eou/valid_options_no_eou.npy'


        # Locations of split sentences of training/valdiation data
        self.train_context_no_eou_path = 'train_data/train_advising_split/train_context_split.npy'
        self.train_target_no_eou_path = 'train_data/train_advising/train_target.npy'
        self.train_options_no_eou_path = 'train_data/train_advising_split/train_options_split.npy'

        self.valid_context_no_eou_path = 'valid_data/valid_advising_split/valid_context_split.npy'
        self.valid_target_no_eou_path = 'valid_data/valid_advising/valid_target.npy'
        self.valid_options_no_eou_path = 'valid_data/valid_advising_split/valid_options_split.npy'


def create_data_locations():
    return data_locations()
