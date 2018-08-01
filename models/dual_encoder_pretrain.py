import models.helpers as helpers

import numpy as np
import keras

from keras import backend as K
from keras.layers import Dense, TimeDistributed, Activation, LSTM, Embedding, Reshape, Lambda, Permute, NonMasking

def get_embeddings(hparams):

    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    print("vocab_array / dict loaded.")
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path,
                                                            vocab=set(vocab_array))
    print("glove_vectors / dict loaded.")
    W = helpers.build_initial_embedding_matrix(vocab_dict,
                                                glove_dict,
                                                glove_vectors,
                                                hparams.de_embedding_dim)
    print("Embedding matrix built.")
    return W



def dual_encoder_model(hparams, context, utterances):

    print("context_shape: ", context.shape)
    print("utterances_shape: ", utterances.shape)

    # Initialize embeddings randomly or with pre-trained vectors
    # embeddings_W = get_embeddings(hparams)
    embeddings_W = np.load('data/embedding_W.npy')
    print("embeddings_W: ", embeddings_W.shape)
    
    # Define embedding layer shared by context and 100 utterances
    embedding_context_layer = Embedding(input_dim=4899,
                            output_dim=hparams.de_embedding_dim,
                            weights=[embeddings_W],
                            input_length=hparams.max_context_len,
                            mask_zero=True,
                            trainable=False)

    # Context Embedding (Output shape: BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    context_embedded = embedding_context_layer(context)
    # context_embedded = Masking()(context_embedded)
    print("context_embedded: ", context_embedded.shape)
    print("context_embedded (history): ", context_embedded._keras_history)

    # Utterances Embedding (Output shape: NUM_OPTIONS(100) x BATCH_SIZE(?) x LEN_SEQ(160) x EMBEDDING_DIM(300))
    embedding_utterance_layer = Embedding(input_dim=4899,
                            output_dim=hparams.de_embedding_dim,
                            weights=[embeddings_W],
                            input_length=hparams.max_utterance_len,
                            mask_zero=True,
                            trainable=False)

    utterances_embedded = TimeDistributed(embedding_utterance_layer,
                                            input_shape=(hparams.num_utterance_options, hparams.max_utterance_len))(utterances)
    print("Utterances_embedded: ", utterances_embedded.shape)
    print("Utterances_embedded (history): ", utterances_embedded._keras_history)

    # Define LSTM Context encoder
    LSTM_context = LSTM(hparams.de_rnn_dim,
                        input_shape=(hparams.max_context_len, hparams.de_embedding_dim),
                        unit_forget_bias=True,
                        return_state=False,
                        return_sequences=False)

    # Encode context (Output shape: BATCH_SIZE(?) x RNN_DIM(256))
    context_encoded_h = LSTM_context(context_embedded)
    context_encoded_h = NonMasking()(context_encoded_h)
    print("context_encoded_h: ", context_encoded_h.shape)
    print("context_encoded_h (history): ", context_encoded_h._keras_history)
    
        
    # Define LSTM Utterances encoder
    LSTM_utterances = LSTM(hparams.de_rnn_dim,
                        input_shape=(hparams.max_utterance_len, hparams.de_embedding_dim),
                        unit_forget_bias=True,
                        return_state=False,
                        return_sequences=False)
    
    # Encode utterances (100 times, Output shape: BATCH_SIZE(?) x NUM_OPTIONS(100) x RNN_DIM(256) )
    all_utterances_encoded_h = TimeDistributed(LSTM_utterances,
                                                input_shape=(hparams.num_utterance_options, hparams.max_utterance_len, hparams.de_embedding_dim))(utterances_embedded)

    print("all_utterances_encoded_h: ", all_utterances_encoded_h.shape)
    all_utterances_encoded_h = NonMasking()(all_utterances_encoded_h)
    all_utterances_encoded_h = Permute((2,1))(all_utterances_encoded_h)
    print("all_utterances_encoded_h: ", all_utterances_encoded_h.shape)
    print("all_utterances_encoded_h: ", all_utterances_encoded_h._keras_history)


    # Generate (expected) response from context: C_transpose * M
    # (Output shape: BATCH_SIZE(?) x RNN_DIM(256))
    matrix_multiplication_layer = Dense(hparams.de_rnn_dim,
                                        use_bias=True,
                                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=None))
    generated_response = matrix_multiplication_layer(context_encoded_h)
    print("genearted_response: ", generated_response.shape)
    print("history555555: ", generated_response._keras_history)

    # (Output shape: BATCH_SIZE(?) x RNN_DIM(256) -> BATCH_SIZE(?) x EXTRA_DIM(1) x RNN_DIM(256))
    generated_response = Reshape((1,hparams.de_rnn_dim))(generated_response)
    print("genearted_response_expand_dims: ", generated_response.shape)
    print("genearted_response_expand_dims: ", generated_response._keras_history)
    

    # Dot product between generated response and each of 100 utterances(actual response r): C_transpose * M * r
    # (Output shape: BATCH_SIZE(?) x EXTRA_DIM(1) x NUM_OPTIONS(100))
    batch_matrix_multiplication_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    logits = batch_matrix_multiplication_layer([generated_response, all_utterances_encoded_h])
    print("logits: ", logits.shape)
    print("logtis: ", logits._keras_history)

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
