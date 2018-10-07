import models.helpers as helpers

import numpy as np
import keras

from keras import backend as K
from keras.layers import Dense, TimeDistributed, Activation, LSTM, Bidirectional, Dropout, Masking, RepeatVector
from keras.layers import Embedding, Reshape, Lambda, Permute, NonMasking, Add, Dot, Multiply, Concatenate


def memLstm_custom_model(hparams,
                            context, context_mask,
                            utterances,
                            context_profile_flag, utterances_profile_flag):

    print("context: ", context._keras_shape)
    print("context_mask: ", context_mask._keras_shape)
    print("utterances: ", utterances._keras_shape)
    print("context_profile_flag: ", context_profile_flag._keras_shape)
    print("utterances_profile_flag: ", utterances_profile_flag._keras_shape)
    # print("profile_mask: ", profile_mask._keras_shape)

    # Use embedding matrix pretrained by Gensim
    # embeddings_W = np.load('data/advising/wiki_advising_embedding_W.npy')
    embeddings_W = np.load('/ext2/dstc7/data/wiki_advising_aKB_test1_test2_embedding_W.npy')
    print("embeddings_W: ", embeddings_W.shape)
    

    ################################## Define Regular Layers ##################################
    # Utterances Embedding (Output shape: NUM_OPTIONS(100) x BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    embedding_context_layer = Embedding(input_dim=hparams.vocab_size,
                                        output_dim=hparams.memn2n_embedding_dim,
                                        weights=[embeddings_W],
                                        input_length=hparams.max_context_len,
                                        mask_zero=True,
                                        trainable=False)
    
    embedding_utterance_layer = Embedding(input_dim=hparams.vocab_size,
                                            output_dim=hparams.memn2n_embedding_dim,
                                            weights=[embeddings_W],
                                            input_length=hparams.max_utterance_len,
                                            mask_zero=True,
                                            trainable=False)

    # Define LSTM Context encoder 1
    LSTM_A = LSTM(hparams.memn2n_rnn_dim,
                    input_shape=(hparams.max_context_len, hparams.memn2n_embedding_dim),
                    use_bias=True,
                    unit_forget_bias=True,
                    return_state=True,
                    return_sequences=True)

    # Define LSTM Utterances encoder
    LSTM_B = LSTM(hparams.memn2n_rnn_dim,
                    input_shape=(hparams.max_utterance_len, hparams.memn2n_embedding_dim),
                    use_bias=True,
                    unit_forget_bias=True,
                    return_state=False,
                    return_sequences=False)
    '''
    # Define LSTM Context encoder 2
    LSTM_C = LSTM(hparams.memn2n_rnn_dim,
                        input_shape=(hparams.max_context_len, hparams.memn2n_embedding_dim+2),
                        unit_forget_bias=True,
                        return_state=False,
                        return_sequences=True)
    '''

    # Define Dense layer to transform utterances
    Matrix_utterances = Dense(hparams.memn2n_rnn_dim,
                                use_bias=False,
                                kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                                        stddev=1.0,
                                                                                        seed=None),
                                input_shape=(hparams.memn2n_rnn_dim,))
    
    # Define Dense layer to do softmax
    Dense_2 = Dense(1,
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                            stddev=1.0,
                                                                            seed=None),
                            input_shape=(hparams.memn2n_rnn_dim,))
    
    ################################## Define Custom Layers ##################################
    # Define max layer
    max_layer = Lambda(lambda x: K.max(x, axis=-1))

    # Define repeat element layer
    custom_repeat_layer = Lambda(lambda x: K.repeat_elements(x, hparams.max_context_len, 1))
    custom_repeat_layer2= Lambda(lambda x: K.repeat_elements(x, hparams.num_utterance_options, 1))

    # Expand dimension layer
    expand_dim_layer = Lambda(lambda x: K.expand_dims(x, axis=1))

    # Amplify layer
    amplify_layer = Lambda(lambda x: x*hparams.amplify_val)

    # Define Softmax layer
    softmax_layer = Lambda(lambda x: K.softmax(Masking()(x), axis=-1))
    softmax_layer2= Lambda(lambda x: K.softmax(Masking()(x), axis=1))

    # Define Stack & Concat layers
    Stack = Lambda(lambda x: K.stack(x, axis=1))

    # Naming tensors
    responses_dot_layer = Lambda(lambda x: x, name='responses_dot')
    responses_attention_layer = Lambda(lambda x: x, name='responses_attention')
    context_attention_layer = Lambda(lambda x: x, name='context_attention')

    # Concat = Lambda(lambda x: K.concatenate(x, axis=1))
    
    # Sum up last dimension
    Sum = Lambda(lambda x: K.sum(x, axis=-1))
    Sum2= Lambda(lambda x: K.sum(x, axis=1))

    # Normalize layer
    Normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    # Define tensor slice layer
    GetFirstHalfTensor = Lambda(lambda x: x[:, :, :hparams.memn2n_rnn_dim])
    GetFirstTensor = Lambda(lambda x: x[:, 0, :])
    GetLastHalfTensor = Lambda(lambda x: x[:, :, hparams.memn2n_rnn_dim:])
    GetLastTensor = Lambda(lambda x: x[:, -1, :])

    GetReverseTensor = Lambda(lambda x: K.reverse(x, axes=1))

    ################################## Apply layers ##################################
    # Prepare Masks
    utterances_mask = Reshape((1,hparams.max_context_len))(context_mask)
    utterances_mask = custom_repeat_layer2(utterances_mask)
    context_mask = Reshape((hparams.max_context_len,1))(context_mask)

    # Prepare Profile
    # context_profile_flag_max = max_layer(context_profile_flag)
    # context_profile_flag_max = Reshape((hparams.max_context_len,1))(context_profile_flag_max)

    # Context Embedding: (BATCH_SIZE(?) x CONTEXT_LEN x EMBEDDING_DIM)
    context_embedded = embedding_context_layer(context)
    print("context_embedded: ", context_embedded.shape)
    print("context_embedded (history): ", context_embedded._keras_history, '\n')
    # Skip this?
    # context_embedded = Concatenate(axis=-1)([context_embedded, context_speaker])


    # Utterances Embedding: (BATCH_SIZE(?) x NUM_OPTIONS x UTTERANCE_LEN x EMBEDDING_DIM)
    utterances_embedded = TimeDistributed(embedding_utterance_layer,
                                            input_shape=(hparams.num_utterance_options,
                                                        hparams.max_utterance_len))(utterances)
    print("Utterances_embedded: ", utterances_embedded.shape)
    print("Utterances_embedded (history): ", utterances_embedded._keras_history, '\n')



    # Encode context A: (BATCH_SIZE(?) x CONTEXT_LEN x RNN_DIM)
    all_context_encoded_Forward,\
    all_context_encoded_Forward_h,\
    all_context_encoded_Forward_c = LSTM_A(context_embedded)

    all_context_encoded_Backward,\
    all_context_encoded_Backward_h,\
    all_context_encoded_Backward_c = LSTM_A(Masking()(GetReverseTensor(context_embedded)))#,
                                            #initial_state=[all_context_encoded_Forward_h, all_context_encoded_Forward_c])
    all_context_encoded_Backward = Masking()(GetReverseTensor(all_context_encoded_Backward))

    # print("context_encoded_A: ", len(context_encoded_A))
    print("all_context_encoded_Forward: ", all_context_encoded_Forward.shape)
    print("all_context_encoded_Forward (history): ", all_context_encoded_Forward._keras_history)
    print("all_context_encoded_Backward: ", all_context_encoded_Backward.shape)
    print("all_context_encoded_Backward (history): ", all_context_encoded_Backward._keras_history)

    # Define bi-directional
    all_context_encoded_Bidir_sum = Add()([all_context_encoded_Forward,
                                            all_context_encoded_Backward])
    
    print("all_context_encoded_Bidir_sum: ", all_context_encoded_Bidir_sum._keras_shape)
    print("all_context_encoded_Bidir_sum: (history)", all_context_encoded_Bidir_sum._keras_history, '\n')
 

    prof_aug_context_encoded_Forward = Concatenate(axis=-1)([all_context_encoded_Forward,
                                                                context_profile_flag])
    prof_aug_context_encoded_Backward= Concatenate(axis=-1)([all_context_encoded_Backward,
                                                                context_profile_flag])
    print("prof_aug_context_encoded_Forward: ", prof_aug_context_encoded_Forward.shape)
    print("prof_aug_context_encoded_Forward (history): ", prof_aug_context_encoded_Forward._keras_history)
    print("prof_aug_context_encoded_Backward: ", prof_aug_context_encoded_Backward.shape)
    print("prof_aug_context_encoded_Backward (history): ", prof_aug_context_encoded_Backward._keras_history, '\n')



    # Encode utterances B: (BATCH_SIZE(?) x NUM_OPTIONS(100) x RNN_DIM)
    all_utterances_encoded_B = TimeDistributed(LSTM_B,
                                                input_shape=(hparams.num_utterance_options,
                                                            hparams.max_utterance_len,
                                                            hparams.memn2n_embedding_dim))(utterances_embedded)
    all_utterances_encoded_B = TimeDistributed(Matrix_utterances,
                                                input_shape=(hparams.num_utterance_options,
                                                            hparams.memn2n_rnn_dim))(all_utterances_encoded_B)
    print("all_utterances_encoded_B: ", all_utterances_encoded_B._keras_shape)
    print("all_utterances_encoded_B: (history)", all_utterances_encoded_B._keras_history, '\n')


    responses_attention = []
    responses_dot = []
    context_attention = None
    for i in range(hparams.hops):
        print(str(i+1) + 'th hop:')
        # 1st Attention & Weighted Sum
        # between Utterances_B(NUM_OPTIONS x RNN_DIM) and Contexts_encoded_Forward(CONTEXT_LEN x RNN_DIM)
        # and apply Softmax
        # (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100) x CONTEXT_LEN)
        
        prof_aug_utterances_encoded_B = Concatenate(axis=-1)([all_utterances_encoded_B,
                                                                utterances_profile_flag])
        attention_Forward = Dot(axes=[2,2])([prof_aug_utterances_encoded_B,
                                                prof_aug_context_encoded_Forward])
        dot_Forward = attention_Forward
        attention_Forward = amplify_layer(attention_Forward)
        attention_Forward = Add()([attention_Forward, utterances_mask])
        attention_Forward = softmax_layer(attention_Forward)
        print("attention_Forward: ", attention_Forward._keras_shape)
        print("attention_Forward: (history)", attention_Forward._keras_history)

        # between Attention(NUM_OPTIONS x CONTEXT_LEN) and Contexts_A(CONTEXT_LEN x RNN_DIM)
        # equivalent to weighted sum of Contexts_A according to Attention
        # (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100) x RNN_DIM)
        weighted_sum_Forward = Dot(axes=[2,1])([attention_Forward,
                                                    all_context_encoded_Bidir_sum])
        print("weighted_sum: ", weighted_sum_Forward._keras_shape)
        print("weighted_sum: (history)", weighted_sum_Forward._keras_history, '\n')

        # (Output shape: ? x NUM_OPTIONS(100) x RNN_DIM)
        all_utterances_encoded_B = Add()([weighted_sum_Forward, all_utterances_encoded_B])


        # 2nd Attention & Weighted Sum
        # between Utterances_B(NUM_OPTIONS x RNN_DIM) and Contexts_encoded_Backward(CONTEXT_LEN x RNN_DIM)
        # and apply Softmax
        # (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100) x CONTEXT_LEN)
        prof_aug_utterances_encoded_B = Concatenate(axis=-1)([all_utterances_encoded_B,
                                                                utterances_profile_flag])

        attention_Backward = Dot(axes=[2,2])([prof_aug_utterances_encoded_B,
                                                prof_aug_context_encoded_Backward])
        dot_Backward = attention_Backward
        attention_Backward = amplify_layer(attention_Backward)
        attention_Backward = Add()([attention_Backward, utterances_mask])
        attention_Backward = softmax_layer(attention_Backward)
        print("attention_Backward: ", attention_Backward._keras_shape)
        print("attention_Backward: (history)", attention_Backward._keras_history)

        # between Attention(NUM_OPTIONS x CONTEXT_LEN) and Contexts_A(CONTEXT_LEN x RNN_DIM)
        # equivalent to weighted sum of Contexts_A according to Attention
        # (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100) x RNN_DIM)
        weighted_sum_Backward = Dot(axes=[2,1])([attention_Backward,
                                                    all_context_encoded_Bidir_sum])
        print("weighted_sum_Backward: ", weighted_sum_Backward._keras_shape)
        print("weighted_sum_Backward: (history)", weighted_sum_Backward._keras_history, '\n')

        # (Output shape: ? x NUM_OPTIONS(100) x RNN_DIM)
        all_utterances_encoded_B = Add()([weighted_sum_Backward, all_utterances_encoded_B])


        dot_Forward = Reshape((1,hparams.num_utterance_options,hparams.max_context_len))(dot_Forward)
        dot_Backward= Reshape((1,hparams.num_utterance_options,hparams.max_context_len))(dot_Backward)
        att_Forward = expand_dim_layer(attention_Forward)
        att_Backward= expand_dim_layer(attention_Backward)

        merge_dots = Concatenate(axis=1)([dot_Forward,
                                            dot_Backward])
        merge_responses = Concatenate(axis=1)([att_Forward,
                                                att_Backward])
        responses_dot.append(merge_dots)
        responses_attention.append(merge_responses)

        print("repsonses_attention[i]:", merge_responses._keras_shape)


        if i < hparams.hops-1:
            continue
            '''
            temp = all_context_encoded_Forward
            all_context_encoded_Forward = all_context_encoded_Backward
            all_context_encoded_Backward = temp
            '''
        else:
            print("hop ended")

            ############# Attention to Context #############
            # (Output shape: ? x MAX_CONTEXT_LEN x 1)
            attention_Forward_wrt_context =\
            TimeDistributed(Dense_2,
                            input_shape=(hparams.max_context_len,
                                        hparams.memn2n_rnn_dim))(all_context_encoded_Forward)
            attention_Forward_wrt_context = amplify_layer(attention_Forward_wrt_context)
            attention_Forward_wrt_context = Add()([attention_Forward_wrt_context,
                                                    context_mask])
            attention_Forward_wrt_context = softmax_layer2(attention_Forward_wrt_context)

            # (Output shape: ? x 1 x RNN_DIM)
            weighted_sum_Forward_wrt_context = Dot(axes=[1,1])([attention_Forward_wrt_context,
                                                                    all_context_encoded_Bidir_sum])


            # (Output shape: ? x MAX_CONTEXT_LEN x 1)
            attention_Backward_wrt_context =\
            TimeDistributed(Dense_2,
                            input_shape=(hparams.max_context_len,
                                        hparams.memn2n_rnn_dim))(all_context_encoded_Backward)
            attention_Backward_wrt_context = amplify_layer(attention_Backward_wrt_context)
            attention_Backward_wrt_context = Add()([attention_Backward_wrt_context,
                                                    context_mask])
            attention_Backward_wrt_context = softmax_layer2(attention_Backward_wrt_context)

            # (Output shape: ? x 1 x RNN_DIM)
            weighted_sum_Backward_wrt_context = Dot(axes=[1,1])([attention_Backward_wrt_context,
                                                                    all_context_encoded_Bidir_sum])

            att_Forward_wrt_context = Reshape((1,hparams.max_context_len))(attention_Forward_wrt_context)
            att_Backward_wrt_context= Reshape((1,hparams.max_context_len))(attention_Backward_wrt_context)
            context_attention = Concatenate(axis=1)([att_Forward_wrt_context,
                                                    att_Backward_wrt_context])


            context_encoded_AplusC = Add()([weighted_sum_Forward_wrt_context,
                                            weighted_sum_Backward_wrt_context])
            print("context_encoded_AplusC: ", context_encoded_AplusC.shape)
            print("context_encoded_AplusC: (history)", context_encoded_AplusC._keras_history, '\n')

            # (Output shape: ? x 1 x NUM_OPTIONS(100))
            logits = Dot(axes=[2,2])([context_encoded_AplusC, all_utterances_encoded_B])
            logits = Reshape((hparams.num_utterance_options,))(logits)
            print("logits: ", logits.shape)
            print("logits: (history)", logits._keras_history, '\n')


            # Softmax layer for probability of each of Dot products in previous layer
            # Softmaxing logits (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100))
            probs = Activation('softmax', name='probs')(logits)
            print("probs: ", probs.shape)
            print("final History: ", probs._keras_history, '\n')

    # Return probabilities(likelihoods) of each of utterances
    # Those will be used to calculate the loss ('sparse_categorical_crossentropy')
    if hparams.hops == 1:
        responses_dot = Reshape((1,2,hparams.num_utterance_options,hparams.max_context_len))(responses_dot[0])
        responses_attention = Reshape((1,2,hparams.num_utterance_options,hparams.max_context_len))(responses_attention[0])

    else:
        responses_dot = Stack(responses_dot)
        responses_attention = Stack(responses_attention)


    responses_dot = responses_dot_layer(responses_dot)
    responses_attention = responses_attention_layer(responses_attention)
    context_attention = context_attention_layer(context_attention)
    print("repsonses_attention:", responses_attention._keras_shape)
    print("context_attention:", context_attention._keras_shape)
    return probs, context_attention, responses_attention, responses_dot
