import models.helpers as helpers

import numpy as np
import keras

from keras import backend as K
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, TimeDistributed, Dense, Flatten, Activation, Embedding, Reshape, Concatenate, Lambda, Permute, Dropout


def get_embeddings(hparams):

    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    print("vocab_array / dict loaded.")
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path,
                                                            vocab=set(vocab_array))
    print("glove_vectors / dict loaded.")
    W = helpers.build_initial_embedding_matrix(vocab_dict,
                                                glove_dict,
                                                glove_vectors,
                                                hparams.embedding_dim)
    print("Embedding matrix built.")
    return W


def cnn_1d_model(hparams, context, utterances):

    # Initialize embeddings randomly or with pre-trained vectors
    embeddings_W = get_embeddings(hparams)
    print("embeddings_W: ", embeddings_W.shape)
    
    # Define embedding layer shared by context and 100 utterances
    embedding_layer = Embedding(input_dim=hparams.vocab_pool_size,
                            output_dim=hparams.embedding_dim,
                            weights=[embeddings_W],
                            input_length=hparams.max_seq_len,
                            mask_zero=False,
                            trainable=True)

    # Context Embedding (Output shape: BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    context_embedded = embedding_layer(context)
    # context_embedded = Masking()(context_embedded)
    print("context_embedded: ", context_embedded.shape)
    print("context_embedded (history): ", context_embedded._keras_history)

    # Utterances Embedding (Output shape: NUM_OPTIONS(100) x BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    #             -> Utterances_embedded: (?, 100, 160, 300)
    utterances_embedded = TimeDistributed(embedding_layer, input_shape=(hparams.num_utterance_options, hparams.max_seq_len))(utterances)
    print("Utterances_embedded: ", utterances_embedded.shape)
    print("Utterances_embedded (history): ", utterances_embedded._keras_history)


    # Define CNN context & utterances encoders
    context_max_maps=[]
    utterances_max_maps=[]
    for k_s in hparams.kernel_size:
        CNN_1D = Conv1D(filters=hparams.num_filters,
                            kernel_size=k_s,
                            strides=1,
                            padding="valid",
                            activation="relu",
                            input_shape=(hparams.max_seq_len, hparams.embedding_dim))
        
        # Output shape: BATCH_SIZE(?) x (LEN_SEQ - k_s + 1) x NUM_FILTERS (for one kernel_size)
        context_feature_map = CNN_1D(context_embedded)
        
        # Output shape: BATCH_SIZE(?) x 1 x NUM_FILTERS (for one kernel_size)
        context_max_map = MaxPooling1D(pool_size=hparams.max_seq_len-k_s+1)(context_feature_map)
        context_max_maps.append(Flatten()(context_max_map))


        CNN_2D = Conv2D(filters=hparams.num_filters,
                            kernel_size=(1,k_s),
                            strides=1,
                            padding="valid",
                            activation='relu',
                            input_shape=(hparams.num_utterance_options,hparams.max_seq_len,hparams.embedding_dim))
        # Output shape: BATCH_SIZE(?) x NUM_UTTERANCES(100) x (LEN_SEQ - k_s + 1) x NUM_FILTERS (for one kernel_size)
        utterances_feature_map=CNN_2D(utterances_embedded)

        # Output shape: BATCH_SIZE(?) x NUM_UTTERANCES(100) x 1 x NUM_FILTERS (for one kernel_size)
        utterances_max_map = MaxPooling2D(pool_size=(1,hparams.max_seq_len - k_s + 1))(utterances_feature_map)

        # Output shape: BATCH_SIZE(?) x NUM_UTTERANCES(100) x NUM_FILTERS (for one kernel_size)
        Flatten_u_by_u = Reshape((hparams.num_utterance_options, hparams.num_filters))

        utterances_max_maps.append(Flatten_u_by_u(utterances_max_map))

    # Output shape: BATCH_SIZE(?) x (NUM_KERNEL_SIZES x NUM_FILTERS)
    context_encoded = Concatenate()(context_max_maps)
    print("context_encoded: ", context_encoded.shape)
    print("context_encoded: ", context_encoded._keras_history)

    context_encoded = Dropout(hparams.drop_rate)(context_encoded)


    # Output shape: BATCH_SIZE(?) x NUM_UTTERANCES(100) x (NUM_KERNEL_SIZES x NUM_FILTERS)
    #            -> BATCH_SIZE(?) x (NUM_KERNEL_SIZES x NUM_FILTERS) x NUM_UTTERANCES(100)
    all_utterances_encoded = Concatenate(axis=2)(utterances_max_maps)
    all_utterances_encoded = Flatten()(all_utterances_encoded)
    all_utterances_encoded = Dropout(hparams.drop_rate)(all_utterances_encoded)
    all_utterances_encoded = Reshape((hparams.num_utterance_options, len(hparams.kernel_size) * hparams.num_filters))(all_utterances_encoded)
    print("all_utterances_encoded: ", all_utterances_encoded.shape)
    print("all_utterances_encoded: ", all_utterances_encoded._keras_history)
    all_utterances_encoded = Permute((2,1))(all_utterances_encoded)


    # Generate (expected) response from context: C_transpose * M
    # (Output shape: BATCH_SIZE(?) x (NUM_KERNEL_SIZES x NUM_FILTERS)
    matrix_multiplication_layer = Dense(len(hparams.kernel_size)*hparams.num_filters,
                                        use_bias=False,
                                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=None))
    generated_response = matrix_multiplication_layer(context_encoded)
    print("genearted_response: ", generated_response.shape)
    print("history555555: ", generated_response._keras_history)

    # (Output shape: BATCH_SIZE(?) x 1 x (NUM_KERNEL_SIZES x NUM_FILTERS)
    generated_response = Reshape((1, len(hparams.kernel_size)*hparams.num_filters))(generated_response)
    print("genearted_response_expand_dims: ", generated_response.shape)
    print("genearted_response_expand_dims: ", generated_response._keras_history)
    

    # Dot product between generated response and each of 100 utterances(actual response r): C_transpose * M * r
    # (Output shape: BATCH_SIZE(?) x EXTRA_DIM(1) x NUM_OPTIONS(100))
    batch_matrix_multiplication_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    logits = batch_matrix_multiplication_layer([generated_response, all_utterances_encoded])
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