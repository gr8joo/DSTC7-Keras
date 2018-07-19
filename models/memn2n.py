# import tensorflow as tf
# import utils.MaskEatingLambda as MEL

import models.helpers as helpers

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, TimeDistributed, Activation, LSTM, Input, Reshape, concatenate, Permute, dot, add, Dropout, Add
from keras.layers import LSTM, TimeDistributed, RepeatVector, Multiply, Dot, multiply
from keras.layers.core import *
from keras.activations import softmax
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
                                                hparams.memn2n_embedding_dim)
    print("Embedding matrix built.")
    return W


def memn2n_model(hparams, context, utterances):


    input_encoder_m = Embedding(input_dim = hparams.vocab_size,
                                output_dim = hparams.memn2n_embedding_dim,
                                mask_zero = True,
                                trainable = False)

    input_encoder_c = Embedding(input_dim = hparams.vocab_size,
                                output_dim = hparams.query_maxlen,
                                mask_zero = True,
                                trainable = False)

    question_encoder = Embedding(input_dim = hparams.vocab_size,
                                 output_dim = hparams.memn2n_embedding_dim,
                                 input_length = hparams.query_maxlen,
                                 mask_zero = True,
                                 trainable = False)

    LSTM_answer = LSTM(hparams.memn2n_rnn_dim)
    Dense1 = Dense(hparams.dense1)
    Dense2 = Dense(hparams.dense2)
    Dropout_ = Dropout(hparams.memn2n_drop_rate)
    Softmax = Activation('softmax')

    for hop in range(hparams.hops):


        input_encoded_m = input_encoder_m(context)                                   # (batch, story_maxlen, embedding_dim)
        input_encoded_m = Dropout(hparams.memn2n_drop_rate)(input_encoded_m)
        input_encoded_m = NonMasking()(input_encoded_m)
        input_encoded_c = input_encoder_c(context)                                   # (batch, story_maxlen, query_maxlen)
        input_encoded_c = Dropout(hparams.memn2n_drop_rate)(input_encoded_c)
        input_encoded_c = NonMasking()(input_encoded_c)
        
        if hop == 0:
            question_encoded = TimeDistributed(question_encoder)(utterances)         # (batch, num_utterance_options, query_maxlen, embedding_dim)
            question_encoded = NonMasking()(question_encoded)
        else:
            question_encoded = answer


        #print(input_encoded_m.shape)
        #print(question_encoded.shape)

        reshape_dim = hparams.story_maxlen * hparams.memn2n_embedding_dim
        input_encoded_m = Reshape((1,reshape_dim))(input_encoded_m)                  # (batch, 1, story_maxlen * embedding_dim)
        #print(" after reshape input_encoded_m.shape : ",input_encoded_m.shape)
        custom_repeat_element = Lambda(lambda x: K.repeat_elements(x, hparams.num_utterance_options, 1))
        input_encoded_m = custom_repeat_element(input_encoded_m)
        #print(input_encoded_m.shape)
        input_encoded_m = Reshape(( hparams.num_utterance_options, hparams.story_maxlen, hparams.memn2n_embedding_dim, ))(input_encoded_m)  # (batch, num_utterance_options, story_maxlen, embedding_dim)
        #print(input_encoded_m.shape)


        batch_matmul = Lambda(lambda x: K.batch_dot(x[0],x[1]))
        


        #print( "input_encoded_m._keras_shape:",input_encoded_m._keras_shape)
        #print( "question_encoded._keras_shape:", question_encoded._keras_shape)
        match = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3,3]))([input_encoded_m, question_encoded])   # (batch, num_utterance_options, story_maxlen, query_maxlen)

        #print("match.shape : ",match.shape)                     # (?, 100, 160, 160)
        #print("input_encoded_c.shape: ",input_encoded_c.shape)  # (?, 160, 160)

        reshape_dim2 = hparams.story_maxlen * hparams.query_maxlen
        input_encoded_c = Reshape((1,reshape_dim2,))(input_encoded_c)
        input_encoded_c = custom_repeat_element(input_encoded_c)
        input_encoded_c = Reshape(( hparams.num_utterance_options, hparams.story_maxlen, hparams.query_maxlen, ))(input_encoded_c) # (batch, num_utterance_options, story_maxlen, query_maxlen)

        #print(match.shape)                # (?, 100, 160, 160)
        #print(input_encoded_c.shape)      # (?, 100, 160, 160)
        response = multiply([match, input_encoded_c]) # (batch, num_utterance_options, story_maxlen, query_maxlen)
        #print(response.shape)
        #print(question_encoded.shape)
        answer = concatenate([response, question_encoded], axis=3) # (batch, num_utterance_options, story_maxlen, query_maxlen + embedding_dim )
        #print(answer.shape)


        if hop != hparams.hops-1:



          
            answer = TimeDistributed( Dense2, input_shape=(hparams.story_maxlen , hparams.query_maxlen + hparams.memn2n_embedding_dim) )(answer) #(batch, num_utterance_options, query_maxlen, embedding_dim) 
            #print("answer.shape : ",answer.shape)


        if hop == hparams.hops-1:
            #print(answer.shape)                                       #(?, 100, 160, 310)
            answer = Masking()(answer)
            answer = TimeDistributed( LSTM_answer )(answer)
            answer = TimeDistributed(Dropout_)(answer)
            answer = TimeDistributed(Dense1)(answer)
            answer = NonMasking()(answer)
            answer = Reshape((hparams.num_utterance_options,))(answer)
            probs = Activation('softmax', name='probs')(answer)

    return probs


