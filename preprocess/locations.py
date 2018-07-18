class data_locations:
    def __init__(self):

        # Locations of digitized sentences of training/valdiation data
        self.train_context_path = 'train_data/train_context.npy'# 'train_data/train_context_10.npy'#
        self.train_context_len_path = 'train_data/train_context_len.npy'# 'train_data/train_context_len_10.npy'#
        self.train_target_path = 'train_data/train_target.npy'# 'train_data/train_target_10.npy'#
        self.train_options_path = 'train_data/train_options.npy'# 'train_data/train_options_10.npy'#
        self.train_options_len_path = 'train_data/train_options_len.npy'# 'train_data/train_options_len_10.npy'#

        self.valid_context_path = 'valid_data/valid_context.npy'# 'valid_data/valid_context_10.npy'#
        self.valid_context_len_path = 'valid_data/valid_context_len.npy'# 'valid_data/valid_context_len_10.npy'#
        self.valid_target_path = 'valid_data/valid_target.npy'# 'valid_data/valid_target_10.npy'#
        self.valid_options_path = 'valid_data/valid_options.npy'# 'valid_data/valid_options_10.npy'#
        self.valid_options_len_path = 'valid_data/valid_options_len.npy'# 'valid_data/valid_options_len_10.npy'#

def create_data_locations():
    return data_locations()
