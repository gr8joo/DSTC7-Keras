class hyper_parameters:
    def __init__(self):

        # General parameters
        self.vocab_size = 4900#4899# 321618
        self.num_utterance_options = 100
        
        self.max_seq_len = 90
        self.num_uttr_context = 42

        self.max_context_len = 400
        self.max_utterance_len = 100


        #### Model speicific parameters ####
        # Dual encoder
        self.de_rnn_dim = 256
        self.de_embedding_dim = 300

        # Memory network(n2n)
        self.memn2n_rnn_dim = 128
        self.memn2n_embedding_dim = 300
        self.story_maxlen = 400
        self.query_maxlen = 100
        self.memn2n_drop_rate = 0.3
        self.dense1 = 1
        self.dense2 = 150
        self.hops = 3

        # CNN_1d
        self.cnn_embedding_dim = 300
        self.kernel_size = [2,3,4,5,6,7,8,9,10,11]
        self.num_filters = 20
        # self.cnn_drop_rate = 0.5


        # Optimizer parameters
        self.learning_rate = 0.001
        self.clip_norm = 10.0
        self.batch_size = 40
        # self.eval_batch_size = 10
        self.num_epochs = 100

        # Locations of vocab sets
        self.glove_path = 'data/glove.42B.300d.txt'
        self.vocab_path = 'data/advising/advising.scenario-1.txt'# ubuntu_subtask_1.txt'
        ### self.glove_vectors_path = 'data/glove_vectors.npy'
        ### self.glove_dict_path = 'data/glove_dict.npy'

        ########## Data Variants ##########
        # Locations of digitized sentences of training/valdiation data
        self.train_context_path = 'train_data/train_context.npy'#  'valid_data/valid_context.npy'#
        self.train_context_len_path = 'train_data/train_context_len.npy'# 'valid_data/valid_context_len.npy'#
        self.train_target_path = 'train_data/train_advising/train_target.npy'# 'valid_data/valid_advising/valid_target.npy'#
        self.train_options_path = 'train_data/train_options.npy'# 'valid_data/valid_options.npy'#
        self.train_options_len_path = 'train_data/train_options_len.npy'# 'valid_data/valid_options_len.npy'#

        self.valid_context_path = 'valid_data/valid_context.npy'
        self.valid_context_len_path = 'valid_data/valid_context_len.npy'
        self.valid_target_path = 'valid_data/valid_advising/valid_target.npy'
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
        self.train_context_split_path = 'train_data/train_advising_split/train_context_split.npy'
        self.train_target_split_path = 'train_data/train_advising/train_target.npy'
        self.train_options_split_path = 'train_data/train_advising_split/train_options_split.npy'

        self.valid_context_split_path = 'valid_data/valid_advising_split/valid_context_split.npy'
        self.valid_target_split_path = 'valid_data/valid_advising/valid_target.npy'
        self.valid_options_split_path = 'valid_data/valid_advising_split/valid_options_split.npy'


def create_hyper_parameters():
    return hyper_parameters()
