import models.helpers as helpers
from keras.layers import *
import tensorflow as tf
shape = None

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

def lambda_batch(x):
    global shape
    shape = tf.shape(x)
    return K.reshape(x, (shape[0]*shape[1], x.shape[2], x.shape[3]))

def lambda_batch_output_shape(input_shape):
    global shape
    print (shape)
    assert len(input_shape) == 4
    return tuple([shape[0]*shape[1], input_shape[2], input_shape[3]])

def lambda_back(x):
    global shape
    return K.reshape(x, (shape[0], shape[1], x.shape[1]))

def lambda_back_output_shape(input_shape):
    global shape
    return tuple([None, 100, input_shape[1]])
    # return tuple([shape[0], shape[1], input_shape[1]])

def dual_encoder_model(hparams, context, utterances):
    response_num = 100

    context_embedded = context
    utterances_embedded = utterances

    context_embedded = Masking()(context_embedded)
    utterances_embedded = Masking()(utterances_embedded)

    # Define LSTM Context encoder
    context_encoded_outputs = LSTM(hparams.rnn_dim, return_sequences=False)(context_embedded)
    context_encoded_outputs = NonMasking()(context_encoded_outputs)
    context_encoded_outputs = Reshape((1, hparams.rnn_dim))(context_encoded_outputs)

    # Define LSTM utterances encoder
    utterance_LSTM = LSTM(hparams.rnn_dim, return_sequences=False)

    utterances_embedded = Lambda(lambda_batch, output_shape=lambda_batch_output_shape)(utterances_embedded)
    utterances_encoded_outputs = utterance_LSTM(utterances_embedded)
    # utterances_encoded_outputs = TimeDistributed(utterance_LSTM)(utterances_embedded)
    utterances_encoded_outputs = Lambda(lambda_back, output_shape=lambda_back_output_shape)(utterances_encoded_outputs)
    utterances_encoded_outputs = NonMasking()(utterances_encoded_outputs)

    # Define
    dual_outputs = concatenate([context_encoded_outputs, utterances_encoded_outputs], axis=1) # (None, 101, 200)
    dual_outputs = Dense(200)(dual_outputs)
    dual_outputs = Dense(1)(dual_outputs)
    dual_outputs = Reshape((response_num+1,))(dual_outputs)
    dual_outputs = Dense(response_num)(dual_outputs)
    dual_outputs = Activation('softmax')(dual_outputs)

    return dual_outputs
