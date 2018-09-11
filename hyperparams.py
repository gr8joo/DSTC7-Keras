class hyper_parameters:
    def __init__(self):

        # General parameters
        self.vocab_size =  326385#209825 ubuntu#4900 advising#321618
        self.num_utterance_options = 100
        self.num_knowledge_options = 40
        
        self.max_seq_len = 230#90
        self.num_uttr_context = 42

        self.max_context_len = 600#1250 ubuntu #400 advising
        self.max_utterance_len = 140#230 ubuntu #90 advising
        self.max_knowledge_len = 150
        self.max_profile_len = 24



        #### Model speicific parameters ####
        # Dual encoder
        self.de_rnn_dim = 128#256#300
        self.de_embedding_dim = 300

        # Memory network(n2n)
        self.memn2n_rnn_dim = 128#64
        self.memn2n_embedding_dim = 300
        self.memn2n_drop_rate = 0.3
        self.hops = 3
    
        # CNN_1D
        self.cnn_rnn_dim = 64
        self.cnn_embedding_dim = 300
        self.kernel_size = [2,3,4,5,6,7,8,9,10,11]
        self.num_filters = 10
        # self.cnn_drop_rate = 0.5


        # Optimizer parameters
        self.learning_rate = 0.001#0.0005
        self.clip_norm = 1.0
        self.batch_size = 32
        # self.eval_batch_size = 10
        self.num_epochs = 100

        # Locations of vocab sets
        self.glove_path = 'data/glove.42B.300d.txt'
        self.vocab_path = 'data/advising/advising.scenario-1.txt'# ubuntu_subtask_1.txt'
        ### self.glove_vectors_path = 'data/glove_vectors.npy'
        ### self.glove_dict_path = 'data/glove_dict.npy'

        # Locations of weights
        self.weights_path = 'weights/'+str(self.hops)+'hops'

        ########## Data Variants ##########
        # Locations of digitized sentences of training/valdiation data
        self.train_context_path = 'train_data/train_ubuntu_uKB/train_context.npy'
        self.train_context_len_path = 'train_data/train_ubuntu_uKB/train_context_len.npy'
        self.train_context_speaker_path = 'train_data/train_ubuntu_uKB/train_context_speaker.npy'

        self.train_target_path = 'train_data/train_ubuntu/train_target.npy'
        
        self.train_options_path = 'train_data/train_ubuntu_uKB/train_options.npy'
        self.train_options_len_path = 'train_data/train_ubuntu_uKB/train_options_len.npy'

        self.train_profile_path = 'train_data/train_ubuntu/train_profile.npy'


        self.valid_context_path = 'valid_data/valid_ubuntu_uKB/valid_context.npy'
        self.valid_context_len_path = 'valid_data/valid_ubuntu_uKB/valid_context_len.npy'
        self.valid_context_speaker_path = 'valid_data/valid_ubuntu_uKB/valid_context_speaker.npy'

        self.valid_target_path = 'valid_data/valid_ubuntu/valid_target.npy'

        self.valid_options_path = 'valid_data/valid_ubuntu_uKB/valid_options.npy'
        self.valid_options_len_path = 'valid_data/valid_ubuntu_uKB/valid_options_len.npy'

        self.valid_profile_path = 'valid_data/valid_ubuntu/valid_profile.npy'
        


        # Locations of split sentences of training/valdiation data
        self.train_context_split_path = 'train_data/train_advising_split/train_context_split.npy'
        self.train_target_split_path = 'train_data/train_advising/train_target.npy'
        self.train_options_split_path = 'train_data/train_advising_split/train_options_split.npy'

        self.valid_context_split_path = 'valid_data/valid_advising_split/valid_context_split.npy'
        self.valid_target_split_path = 'valid_data/valid_advising/valid_target.npy'
        self.valid_options_split_path = 'valid_data/valid_advising_split/valid_options_split.npy'


def create_hyper_parameters():
    return hyper_parameters()
