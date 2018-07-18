class hyper_parameters:
    def __init__(self):

        # General parameters
        self.vocab_pool_size = 321618
        self.embedding_dim = 300
        self.num_utterance_options = 100
        self.max_seq_len = 160
        # self.max_context_len = 160
        # self.max_utterance_len = 160
        
        #### Model speicific parameters ####
        # Dual encoder
        self.rnn_dim = 256

        # CNN_1d
        self.kernel_size = [1,2,3,4,5,6]
        self.num_filters = 10
        self.drop_rate = 0.5


        # Optimizer parameters
        self.learning_rate = 0.001
        self.clip_norm = 10.0
        self.batch_size = 40
        # self.eval_batch_size = 10

        # Locations of vocab sets
        self.glove_path = 'data/glove.42B.300d.txt'
        self.vocab_path = 'data/ubuntu_subtask_1.txt'
        ### self.glove_vectors_path = 'data/glove_vectors.npy'
        ### self.glove_dict_path = 'data/glove_dict.npy'

        # Locations of digitized sentences of training/valdiation data
        self.train_context_path = 'train_data/train_context.npy'# 'valid_data/valid_context.npy'#
        self.train_context_len_path = 'train_data/train_context_len.npy'# 'valid_data/valid_context_len.npy'#
        self.train_target_path = 'train_data/train_target.npy'# 'valid_data/valid_target.npy'#
        self.train_options_path = 'train_data/train_options.npy'# 'valid_data/valid_options.npy'#
        self.train_options_len_path = 'train_data/train_options_len.npy'# 'valid_data/valid_options_len.npy'#

        self.valid_context_path = 'valid_data/valid_context.npy'
        self.valid_context_len_path = 'valid_data/valid_context_len.npy'
        self.valid_target_path = 'valid_data/valid_target.npy'
        self.valid_options_path = 'valid_data/valid_options.npy'
        self.valid_options_len_path = 'valid_data/valid_options_len.npy'

def create_hyper_parameters():
    return hyper_parameters()
