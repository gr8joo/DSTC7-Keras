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
    #dataset_vocab_array, dataset_vocab_dict = helpers.load_vocab_w2v(hparams.advising_vocab_path)
    print("vacab_array / dict loaded.")
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path,
                                                            vocab=set(vocab_array))
    #glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.wiki_advising_merged_vector_path,
    #                                                        vocab=set(dataset_vocab_array))


    print("glove_vectors / dict loaded.")
    #W = helpers.build_initial_embedding_matrix(vocab_dict,
    #                                            glove_dict,
    #                                            glove_vectors,
    #                                            hparams.memn2n_embedding_dim)
    print("Embedding matrix built.")
    return W


def memn2n_model(hparams, context, utterances):

    #embeddings_W = get_embeddings(hparams)
    embeddings_W = np.load('data/wiki_advising_merged/embedding_W.npy')

    input_encoder_m = Embedding(input_dim = hparams.vocab_size,
                                output_dim = hparams.memn2n_embedding_dim,
                                weights=[embeddings_W],
                                mask_zero = True,
                                trainable = False)

    input_encoder_c = Embedding(input_dim = hparams.vocab_size,
                                output_dim = hparams.query_maxlen,
                                #weights=[embeddings_W],
                                mask_zero = True,
                                trainable = True)

    question_encoder = Embedding(input_dim = hparams.vocab_size,
                                 output_dim = hparams.memn2n_embedding_dim,
                                 input_length = hparams.query_maxlen,
                                 weights=[embeddings_W],
                                 mask_zero = True,
                                 trainable = False)

    LSTM_answer = LSTM(hparams.memn2n_rnn_dim)
    Dense1 = Dense(hparams.dense1)
    Dense2 = Dense(hparams.memn2n_embedding_dim)
    Dense3 = Dense(hparams.memn2n_embedding_dim)
    Dropout_ = Dropout(hparams.memn2n_drop_rate)
    Softmax = Activation('softmax')
    reshape_dim = hparams.story_maxlen * hparams.memn2n_embedding_dim
    reshape_dim2 = hparams.story_maxlen * hparams.query_maxlen
    custom_repeat_element = Lambda(lambda x: K.repeat_elements(x, hparams.num_utterance_options, 1))

    for hop in range(hparams.hops):


        input_encoded_m = input_encoder_m(context)                                   # (batch, story_maxlen, embedding_dim)
        #input_encoded_m = Dropout(hparams.memn2n_drop_rate)(input_encoded_m)
        input_encoded_m = NonMasking()(input_encoded_m)
        input_encoded_c = input_encoder_c(context)                                   # (batch, story_maxlen, query_maxlen)
        #input_encoded_c = Dropout(hparams.memn2n_drop_rate)(input_encoded_c)
        input_encoded_c = NonMasking()(input_encoded_c)
        
        if hop == 0:
            question_encoded = TimeDistributed(question_encoder)(utterances)         # (batch, num_utterance_options, query_maxlen, embedding_dim)
            question_encoded = NonMasking()(question_encoded)
        else:
            question_encoded = answer

        print("input_encoded_m.shape : ",input_encoded_m.shape) # ( ?, 160, 300 )
        input_encoded_m = Reshape((1,reshape_dim,))(input_encoded_m)                 # (batch, 1, story_maxlen * embedding_dim)
        input_encoded_m = custom_repeat_element(input_encoded_m)                     # (batch, num_utterance_options, story_maxlen * embedding_dim)
        input_encoded_m = Reshape(( hparams.num_utterance_options, hparams.story_maxlen, hparams.memn2n_embedding_dim, ))(input_encoded_m)  # (batch, num_utterance_options, story_maxlen, embedding_dim)
        
        match = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3,3]))([input_encoded_m, question_encoded])   # (batch, num_utterance_options, story_maxlen, query_maxlen)
        #match = Softmax(match)
        #reshape_dim2 = hparams.story_maxlen * hparams.query_maxlen
        input_encoded_c = Reshape((1,reshape_dim2,))(input_encoded_c)               # (batch, 1, story_maxlen * query_maxlen)
        input_encoded_c = custom_repeat_element(input_encoded_c)                    # (batch, num_utterance_options, story_maxlen * query_maxlen)
        input_encoded_c = Reshape(( hparams.num_utterance_options, hparams.story_maxlen, hparams.query_maxlen, ))(input_encoded_c) # (batch, num_utterance_options, story_maxlen, query_maxlen)

        response = multiply([match, input_encoded_c])                               # (batch, num_utterance_options, story_maxlen, query_maxlen)
        #print("response.shape: ",response.shape)
        response = Reshape((hparams.num_utterance_options, hparams.query_maxlen, hparams.story_maxlen,))(response) # (batch, num_utterance_options, query_maxlen, story_maxlen)
        #print("response.shape: ",response.shape)
        
        #######################################
        #instead of concat, let's try add here#
        response = TimeDistributed( Dense3 )(response)
        answer = add([response, question_encoded])                                   # (batch, num_utterance_options, query_maxlen, embedding_dim)




        #answer = concatenate([response, question_encoded], axis=3)                  # (batch, num_utterance_options, query_maxlen, story_maxlen + embedding_dim )
        #answer = concatenate([question_encoded, response], axis=3)

        if hop != hparams.hops-1:

            continue
            #answer = TimeDistributed( Dense2, input_shape=(hparams.query_maxlen , hparams.story_maxlen + hparams.memn2n_embedding_dim) )(answer) #(batch, num_utterance_options, query_maxlen, embedding_dim) 

        if hop == hparams.hops-1:
            answer = Masking()(answer)
            answer = TimeDistributed( LSTM_answer )(answer)                        # (batch, num_utterance_options, 
            """Is LSTM necessary? - Try another layer"""
            #answer = TimeDistributed(Dropout_)(answer)
            answer = TimeDistributed(Dense1)(answer)                                # (batch, num_utterance_options, query_maxlen, 1) 
            answer = NonMasking()(answer)
            print("answer.shape : ",answer.shape)
            answer = Reshape((hparams.num_utterance_options,))(answer)
            probs = Activation('softmax', name='probs')(answer)

    return probs


