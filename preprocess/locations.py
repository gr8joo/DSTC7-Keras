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

        # Locations of STEMMED sentences of training/valdiation data
        self.stem_train_context_path = 'train_data/stem_train_context.npy'
        self.stem_train_context_len_path = 'train_data/stem_train_context_len.npy'
        self.stem_train_target_path = 'train_data/stem_train_target.npy'
        self.stem_train_options_path = 'train_data/stem_train_options.npy'
        self.stem_train_options_len_path = 'train_data/stem_train_options_len.npy'

        self.stem_valid_context_path = 'valid_data/stem_valid_context.npy'
        self.stem_valid_context_len_path = 'valid_data/stem_valid_context_len.npy'
        self.stem_valid_target_path = 'valid_data/stem_valid_target.npy'
        self.stem_valid_options_path = 'valid_data/stem_valid_options.npy'
        self.stem_valid_options_len_path = 'valid_data/stem_valid_options_len.npy'

        # Locations of REDUCED(100->10) sentences of training/valdiation data
        self.red_train_context_path = 'train_data/red_train_context.npy'
        self.red_train_context_len_path = 'train_data/red_train_context_len.npy'
        self.red_train_target_path = 'train_data/red_train_target.npy'
        self.red_train_options_path = 'train_data/red_train_options.npy'
        self.red_train_options_len_path = 'train_data/red_train_options_len.npy'

        self.red_valid_context_path = 'valid_data/red_valid_context.npy'
        self.red_valid_context_len_path = 'valid_data/red_valid_context_len.npy'
        self.red_valid_target_path = 'valid_data/red_valid_target.npy'
        self.red_valid_options_path = 'valid_data/red_valid_options.npy'
        self.red_valid_options_len_path = 'valid_data/red_valid_options_len.npy'

def create_data_locations():
    return data_locations()
