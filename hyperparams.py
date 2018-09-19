class hyper_parameters:
    def __init__(self):

        # General hyperparameters
        self.vocab_size =  333456#326385#209825 ubuntu#4900 advising#321618
        self.num_utterance_options = 100
        self.num_kb_entities = 20
        
        self.max_seq_len = 230#90
        self.num_uttr_context = 42

        self.max_context_len = 600#1250 ubuntu #400 advising
        self.max_utterance_len = 140#230 ubuntu #90 advising
        self.max_kb_len = 240
        self.max_profile_len = 24

        self.neg_inf = -999
        self.amplify_val = 5


        #### Model speicific hyperparameters ####
        # Dual encoder
        self.de_rnn_dim = 128#256#300
        self.de_embedding_dim = 300

        # Memory network(n2n)
        self.memn2n_rnn_dim = 128#64
        self.memn2n_embedding_dim = 300
        self.memn2n_drop_rate = 0.3
        self.hops = 1
    
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
        # self.glove_path = 'data/glove.42B.300d.txt'
        self.vocab_path = '/ext2/dstc7/data/ubuntu_uKB_test_lemma_vocab.txt'
        ### self.glove_vectors_path = 'data/glove_vectors.npy'
        ### self.glove_dict_path = 'data/glove_dict.npy'

        # Locations of weights
        self.weights_path = '/tmp/weights/'+str(self.hops)+'hops'

        ########## Data Variants ##########

        ########## Data Variants ##########
        # Locations of digitized sentences of training/valdiation data
        self.train_context_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_context.npy'#  'valid_data/valid_context.npy'#
        self.train_context_speaker_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_context_speaker.npy'# 'train_data/context_speaker.train'#
        self.train_context_mask_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_context_mask.npy'
        self.train_context_len_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_context_len.npy'# 'valid_data/valid_context_len.npy'#
        
        self.train_target_path = '/ext2/dstc7/train_data/train_ubuntu/train_target.npy'# 'valid_data/valid_advising/valid_target.npy'#

        self.train_options_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_options.npy'# 'valid_data/valid_options.npy'#
        self.train_options_len_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_options_len.npy'# 'valid_data/valid_options_len.npy'#

        self.train_profile_path = '/ext2/dstc7/train_data/train_ubuntu_uKB_test/train_profile.npy'

        self.train_kb_entity_flags_path = '/ext2/dstc7/train_kb_entity_flags.npy'#'train_data/train_ubuntu_uKB_test/train_kb_entity_flags_path'


        self.valid_context_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_context.npy'
        self.valid_context_speaker_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_context_speaker.npy'# 'valid_data/context_speaker.valid'#
        self.valid_context_mask_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_context_mask.npy'
        self.valid_context_len_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_context_len.npy'

        self.valid_target_path = '/ext2/dstc7/valid_data/valid_ubuntu/valid_target.npy'

        self.valid_options_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_options.npy'
        self.valid_options_len_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_options_len.npy'

        self.valid_profile_path = '/ext2/dstc7/valid_data/valid_ubuntu_uKB_test/valid_profile.npy'

        self.valid_kb_entity_flags_path = '/ext2/dstc7/valid_kb_entity_flags.npy'#'valid_data/train_ubuntu_uKB_test/valid_kb_entity_flags_path'

        # Locations of digitized sentences of training/valdiation data
        '''
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
        '''


        # Locations of split sentences of training/valdiation data
        self.train_context_split_path = 'train_data/train_advising_split/train_context_split.npy'
        self.train_target_split_path = 'train_data/train_advising/train_target.npy'
        self.train_options_split_path = 'train_data/train_advising_split/train_options_split.npy'

        self.valid_context_split_path = 'valid_data/valid_advising_split/valid_context_split.npy'
        self.valid_target_split_path = 'valid_data/valid_advising/valid_target.npy'
        self.valid_options_split_path = 'valid_data/valid_advising_split/valid_options_split.npy'


def create_hyper_parameters():
    return hyper_parameters()
