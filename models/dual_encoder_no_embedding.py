# import tensorflow as tf
# import utils.MaskEatingLambda as MEL

import models.helpers as helpers

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, TimeDistributed, Activation, LSTM, Embedding, Input, Reshape, Concatenate, Dot, Lambda, Permute, dot, Masking
from keras.utils import np_utils

def get_embeddings(hparams):

    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    print("vacab_array / dict loaded.")
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path,
                                                            vocab=set(vocab_array))
    print("glove_vectors / dict loaded.")
    W = helpers.build_initial_embedding_matrix(vocab_dict,
                                                glove_dict,
                                                glove_vectors,
                                                hparams.embedding_dim)
    print("Embedding matrix built.")
    return W

def custom_append(x):
  	return x[0].append(x[1])

def dual_encoder_no_embedding_model(hparams, context, utterances):

    # print("context_history: ", context[0]._keras_history)
    # print("utterances_history: ", utterances[0]._keras_history)

    '''
    # Initialize embeddings randomly or with pre-trained vectors
    embeddings_W = get_embeddings(hparams)
    print("embeddings_W: ", embeddings_W.shape)
    
    # Define embedding layer shared by context and 100 utterances
    embedding_layer = Embedding(input_dim=hparams.vocab_pool_size,
                            output_dim=hparams.embedding_dim,
                            weights=[embeddings_W],
                            input_length=hparams.max_sequence_len,
                            mask_zero=True,
                            trainable=True)

    # Context Embedding (Output shape: BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    context_embedded = embedding_layer(context)
    # context_embedded = Masking()(context_embedded)
    print("context_embedded: ", context_embedded.shape)
    print("context_embedded (history): ", context_embedded._keras_history)

    # Utterances Embedding (Output shape: NUM_OPTIONS(100) x BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    utterances_embedded = [embedding_layer(utterances[i])\
                            for i in range(hparams.num_utterance_options)]
    print("Utterances_embedded: ", utterances_embedded[0].shape)
    print("Utterances_embedded (history): ", utterances_embedded[0]._keras_history)
    '''

    # Define LSTM Context encoder
    LSTM_context = LSTM(hparams.rnn_dim,
                        input_shape=(hparams.max_sequence_len, hparams.embedding_dim),
                        unit_forget_bias=True,
                        return_state=True,
                        return_sequences=False)

    # Encode context (Output shape: BATCH_SIZE(?) x RNN_DIM(256))
    context_encoded_outputs,\
    context_encoded_h, context_encoded_c = LSTM_context(context)
    print("context_encoded_h: ", context_encoded_h.shape)
    print("context_encoded_h (history): ", context_encoded_h._keras_history)
    
        
    # Define LSTM Utterances encoder
    LSTM_utterances = LSTM(hparams.rnn_dim,
                        input_shape=(hparams.max_sequence_len, hparams.embedding_dim),
                        unit_forget_bias=True,
                        return_state=True,
                        return_sequences=False)
    
    # Encode utterances (100 times)
    all_utterances_encoded_h = []
    ### transpose_and_expand_dims_layer = Lambda(lambda x: K.expand_dims(K.transpose(x), axis=0))
    ### transpose_layer = Lambda(lambda x: K.transpose(x))
    ### expand_layer = Lambda(lambda x: K.expand_dims(x, axis=1))
    
    for i in range(hparams.num_utterance_options):
        # LSTM layer (Output shape: BATCH_SIZE(?) x RNN_DIM(256) for each of 100 utterances)
        utterances_encoded_outputs,\
        utterances_encoded_h, utterances_encoded_c = LSTM_utterances(utterances[i])

        # Reshape layer (Output shape: BATCH_SIZE(?) x EXTRA_DIM(1) x RNN_DIM(256))
        utterances_encoded_h = Reshape((1,hparams.rnn_dim))(utterances_encoded_h)

        # Permute layer (Output shape: BATCH_SIZE(?) x RNN_DIM(256) x EXTRA_DIM(1))
        utterances_encoded_h = Permute((2,1))(utterances_encoded_h)
        # print("utterances_encoded_h: ", utterances_encoded_h.shape)
        # print("history222222: ", utterances_encoded_h._keras_history)

        # Stacking layer (Output shape: NUM_ITERATIONS x BATCH_SIZE(?) x RNN_DIM(256) x EXTRA_DIM(1))
        all_utterances_encoded_h.append(utterances_encoded_h)
        # print("all_utterances_encoded_h: ", all_utterances_encoded_h.shape)
        # print("history333333: ", all_utterances_encoded_h._keras_history)

    # NUM_OPTIONS x BATCH_SIZE(?) x RNN_DIM(256) x EXTRA_DIM(1) ->  BATCH_SIZE(?) x RNN_DIM(256) x NUM_OPTIONS(100)
    all_utterances_encoded_h = Concatenate(axis=2)(all_utterances_encoded_h)
    print("all_utterances_encoded_h: ", all_utterances_encoded_h.shape)
    print("all_utterances_encoded_h: ", all_utterances_encoded_h._keras_history)


    # Generate (expected) response from context: C_transpose * M
    # (Output shape: BATCH_SIZE(?) x RNN_DIM(256))
    matrix_multiplication_layer = Dense(hparams.rnn_dim,
                                        use_bias=False,
                                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=None))
    generated_response = matrix_multiplication_layer(context_encoded_h)
    print("genearted_response: ", generated_response.shape)
    print("history555555: ", generated_response._keras_history)

    # (Output shape: BATCH_SIZE(?) x RNN_DIM(256) -> BATCH_SIZE(?) x EXTRA_DIM(1) x RNN_DIM(256))
    generated_response = Reshape((1,hparams.rnn_dim))(generated_response)
    print("genearted_response_expand_dims: ", generated_response.shape)
    print("genearted_response_expand_dims: ", generated_response._keras_history)
    

    # Dot product between generated response and each of 100 utterances(actual response r): C_transpose * M * r
    # (Output shape: BATCH_SIZE(?) x EXTRA_DIM(1) x NUM_OPTIONS(100))
    batch_matrix_multiplication_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    logits = batch_matrix_multiplication_layer([generated_response, all_utterances_encoded_h])
    print("logits: ", logits.shape)
    print("logtis: ", logits._keras_history)

    ### squeeze_layer = Lambda(lambda x: K.squeeze(x, 1))
    ### logits = squeeze_layer(logits)
    # Squeezing logits (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100))
    logits = Reshape((hparams.num_utterance_options,), input_shape=(1, hparams.num_utterance_options))(logits)
    print("logits_squeeze: ", logits.shape)
    print("logits_squeeze: ", logits._keras_history)


    # Softmax layer for probability of each of Dot products in previous layer
    # Softmaxing logits (Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100))
    probs = Activation('softmax', name='probs')(logits)
    print("probs: ", probs.shape)
    print("final History: ", probs._keras_history)

    # Return probabilities(likelihoods) of each of utterances
    # Those will be used to calculate the loss ('sparse_categorical_crossentropy')
    return probs
