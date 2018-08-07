import os
import sys
# sys.path.insert(0, '../')
# sys.path.append('../')

import ijson
import functools

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import normalize
from locations import create_data_locations

tf.flags.DEFINE_integer(
    "min_word_frequency", 1, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_string("train_in", None, "Path to input data file")
tf.flags.DEFINE_string("validation_in", None, "Path to validation data file")

tf.flags.DEFINE_string("train_out", None, "Path to output train tfrecords file")
tf.flags.DEFINE_string("validation_out", None, "Path to output validation tfrecords file")

tf.flags.DEFINE_string("advising_vocab_path", None, "Path to save vocabulary processor")
tf.flags.DEFINE_string("entire_vocab_path", None, "Path to save vocabulary txt file")

tf.flags.DEFINE_string("embedding_path", None, "Path to save vocabulary txt file")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.train_in)
VALIDATION_PATH = os.path.join(FLAGS.validation_in)

# ADVISING_VOCAB_PATH = os.path.join(FLAGS.advising_vocab_path)
# ENTIRE_VOCAB_PATH = os.path.join(FLAGS.entire_vocab_path)

MAX_CONTEXT_LEN = 400
MAX_UTTR_NUM = 42
MAX_UTTERANCE_LEN = 90

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def count_len(input_filename):
    max_context_len = 0
    max_utterance_len = 0

    context_len_list = zerolistmaker(10)
    utterance_len_list = zerolistmaker(10)

    max_uttr_num = 0
    max_uttr_len = 0
    with open(input_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line: 
                break
            # line = line.replace('\n', ' ')
            line = line[:-1]
            word = line.split(' ')

            context_len = len(word)
            context_len_list[ int(context_len/100)] += 1
            if context_len > max_context_len:
                max_context_len = context_len

            uttr_num = word.count('__eot__')
            if uttr_num > max_uttr_num:
                max_uttr_num = uttr_num
            uttr_len = 0
            for i in range(uttr_num):
                eot_idx = word.index('__eot__')
                if eot_idx > max_uttr_len:
                    max_uttr_len = eot_idx
                if eot_idx+1 != len(word):
                    word = word[eot_idx+1:]


            for i in range(100):
                line = f.readline()
                # line = line.replace('\n', ' ')
                line = line[:-1]
                word = line.split(' ')
                utterance_len = len(word)
                utterance_len_list[ int(utterance_len/100)] += 1
                if utterance_len > max_utterance_len:
                    max_utterance_len = utterance_len
                ### if utterance_len == 89:
                ###     print(word)
    return context_len_list, utterance_len_list, max_context_len, max_utterance_len, max_uttr_num, max_uttr_len

def create_nparrays(input_filename, vocab):

    with open(input_filename, 'r') as f:
        context_set = []
        utterance_set = []
        while True:
            # Handle context
            line = f.readline()
            if not line: 
                break

            # eliminating '\n'
            line = line[:-1]
            context = line.split(' ')
            
            # elimianating eou
            context = [x for x in context if x != '__eou__']
            
            # replacing eot by eou
            # context = [x if x != '__eot__' else '__eou__' for x in context]
            context_len = len(context)
    
            context_transformed = [vocab.index(context[i])+1 for i in range(context_len)]
            if context_len < MAX_CONTEXT_LEN:
                context_transformed.extend(zerolistmaker(MAX_CONTEXT_LEN-context_len))
            elif context_len > MAX_CONTEXT_LEN:
                context_transformed = context_transformed[context_len-MAX_CONTEXT_LEN:context_len]
            context_set.append(np.array(context_transformed))

            # Handle utterance
            # for i in range(100):
            #     line = f.readline()


            temp_utterance_set = []
            for i in range(100):
                line = f.readline()
                
                # eliminating '\n'
                line = line[:-1]
                utterance = line.split(' ')
                
                # elimiating eou
                utterance = utterance[:-1]
                utterance_len = len(utterance)

                utterance_transformed = [vocab.index(utterance[i])+1 for i in range(utterance_len)]
                if utterance_len < MAX_UTTERANCE_LEN:
                    utterance_transformed.extend(zerolistmaker(MAX_UTTERANCE_LEN-utterance_len))
                elif utterance_len > MAX_UTTERANCE_LEN:
                    # utterance_transformed = utterance_transformed[utterance_len - MAX_UTTERANCE_LEN:utterance_len]
                    utterance_transformed = utterance_transformed[:MAX_UTTERANCE_LEN]
                temp_utterance_set.append(np.array(utterance_transformed))
                ### if utterance_len == 89:
                ###     print(word)
            utterance_set.append(np.array(temp_utterance_set))

    return context_set, utterance_set

def create_nparrays2(input_filename, vocab):

    with open(input_filename, 'r') as f:
        context_set = []
        utterance_set = []
        while True:
            # Handle context
            line = f.readline()
            if not line: 
                break

            # eliminating '\n'
            line = line[:-1]
            context = line.split(' ')
            
            num_uttr_context = context.count('__eou__')
            temp_uttr_set = []
            for i in range(num_uttr_context):
                eou_idx = context.index('__eou__')

                # Including eou
                uttr = context[:eou_idx+1]
                uttr_len = len(uttr)    # equal to eou_idx
                uttr_transformed = [vocab.index(uttr[i])+1 for i in range(uttr_len)]
                if uttr_len < MAX_UTTERANCE_LEN:
                    uttr_transformed.extend(zerolistmaker(MAX_UTTERANCE_LEN-uttr_len))
                elif uttr_len > MAX_UTTERANCE_LEN:
                    uttr_transformed = uttr_transformed[uttr_len - MAX_UTTERANCE_LEN:uttr_len]
                temp_uttr_set.append(np.array(uttr_transformed))

                if eou_idx+2 <= len(context)-1:
                    # Skipping
                    if context[eou_idx+1] == '__eot__':
                        context = context[eou_idx+2:]
                    else:
                        context = context[eou_idx+1:]
                else:
                    if i < num_uttr_context-1:
                        print("Parsing error: Something must be wrong")

            temp_uttr_set = np.concatenate((np.array(temp_uttr_set),
                                np.zeros((MAX_UTTR_NUM-num_uttr_context, MAX_UTTERANCE_LEN,), dtype=int)), axis=0)
            context_set.append(temp_uttr_set)


            temp_utterance_set = []
            for i in range(100):
                line = f.readline()
                
                # eliminating '\n'
                line = line[:-1]
                utterance = line.split(' ')
                
                # elimiating eou
                # utterance = utterance[:-1]
                utterance_len = len(utterance)

                utterance_transformed = [vocab.index(utterance[i])+1 for i in range(utterance_len)]
                if utterance_len < MAX_UTTERANCE_LEN:
                    utterance_transformed.extend(zerolistmaker(MAX_UTTERANCE_LEN-utterance_len))
                elif utterance_len > MAX_UTTERANCE_LEN:
                    # utterance_transformed = utterance_transformed[utterance_len - MAX_UTTERANCE_LEN:utterance_len]
                    utterance_transformed = utterance_transformed[:MAX_UTTERANCE_LEN]
                temp_utterance_set.append(np.array(utterance_transformed))
                ### if utterance_len == 89:
                ###     print(word)
            utterance_set.append(np.array(temp_utterance_set))

    return context_set, utterance_set
def generate_vocab_from_gensim(word_vec_path, vocab_path):
    vocab_size = 0
    vocab = []
    with open(word_vec_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: 
                break
            # line = line.replace('\n', ' ')
            line = line[:-1]
            word = line.split(' ')
            vocab.append(word[0])
    
            '''
            if flag == True:
                str2float = word[1:]
                str2float = [float(str2float[i]) for i in range(len(str2float))]
                embedding_W.append(np.array(str2float))
            else:
                flag = True
            '''
    vocab_size = vocab[0]
    vocab = vocab[1:]
    with open(FLAGS.vocab_path, 'w') as thefile:
        for item in vocab:
            thefile.write("%s\n" % item)

    # embedding_W = np.array(embedding_W)
    # np.save('/home/hjhwang/Codes/dstc7-keras/data/embedding_W.npy', embedding_W)
    
    return vocab, vocab_size

if __name__ == "__main__":

    # Get directories of files to write
    loc = create_data_locations()

    ####### Create advising vocab #######
    advising_vocab = []
    advising_vocab_size = 0
    if FLAGS.advising_vocab_path is not None:
        if os.path.isfile(FLAGS.advising_vocab_path) is False:
            print("Advising Vocab does not exist... Creating to:", FLAGS.advising_vocab_path)
            advising_vocab, advising_vocab_size = generate_vocab_from_gensim('/home/hjhwang/word2vec/advising_lemmatized_vector',
                                                                                FLAGS.advising_vocab_path)
        else:
            print("Advising Vocab exists!")
            f = open(FLAGS.advising_vocab_path,'r')
            A = f.read()
            advising_vocab = A.split('\n')
            advising_vocab_size = len(A)

    ####### Create entire vocab #######
    entire_vocab = []
    entire_vocab_size = 0
    if FLAGS.entire_vocab_path is not None:
        if os.path.isfile(FLAGS.entire_vocab_path) is False:
            print("Entire Vocab does not exist... Creating to", FLAGS.entire_vocab_path)
            entire_vocab, entire_vocab_size = generate_vocab_from_gensim('/home/hjhwang/word2vec/wiki_advising_lemmatized.vector',
                                                                            FLAGS.entire_vocab_path)
        else:
            print("Entire Vocab exists!")
            f = open(FLAGS.entire_vocab_path,'r')
            A = f.read()
            entire_vocab = A.split('\n')
            entire_vocab_size = len(A)

    ####### Extract advising embedding matrix from entire embedding matrix #######
    if FLAGS.embedding_path is not None:
        print("Creating to embedding matirx to", FLAGS.embedding_path)
        entire_embedding_W = np.load('/home/hjhwang/word2vec/wiki_advising_lemmatized.model.trainables.syn1neg.npy')
        print("Entire embedding loaded!")
        embedding_W = [np.zeros(300,)]
        cnt = 0
        for entry in advising_vocab:
            if entry == '':
                break
            elif entry =='cyptography\n':
                print('Hack!!')
            idx = entire_vocab.index(entry)
            # A = entire_embedding_W[idx]/np.linalg.norm(entire_embedding_W[idx])
            x = entire_embedding_W[idx]
            A = normalize(x[:,np.newaxis], axis=0).ravel()
            embedding_W.append(A)

        embedding_W = np.array(embedding_W)
        print("shape of embedding_W: ", embedding_W.shape)
        np.save('/home/hjhwang/Codes/dstc7-keras/data/embedding_W.npy', embedding_W)


    ####### Count len #######
    '''
    print("In Training dataset: (Context / Utterance)")
    context_len_list, utterance_len_list,\
    max_context_len, max_utterance_len,\
    max_uttr_num, max_uttr_len = count_len(TRAIN_PATH)
    for i in range(10):
        print("len less than ",(i+1)*100, ': ', context_len_list[i], ' / ', utterance_len_list[i])
    print("max len: ", max_context_len, ' / ', max_utterance_len)
    print("max number of uttr in Context:", max_uttr_num)
    print("max len of uttr in Context:", max_uttr_len+1)
    print()

    print("In Validation dataset: (Context / Utterance)")
    context_len_list, utterance_len_list,\
    max_context_len, max_utterance_len,\
    max_uttr_num, max_uttr_len = count_len(VALIDATION_PATH)
    for i in range(10):
        print("len less than ",(i+1)*100, ': ', context_len_list[i], ' / ', utterance_len_list[i])
    print("max len: ", max_context_len, ' / ',  max_utterance_len)
    print("max number of uttr in Context:", max_uttr_num)
    print("max len of uttr in Context:", max_uttr_len+1)
    '''

    '''
    ### TODO: Make a function which writes followings to files
    # Create nparray of Training set
    print("Creating np_arrays of Training set...")
    context, options = create_nparrays(input_filename=TRAIN_PATH, vocab=vocab)
    print("Writing np_arrays of Training set...")
    context = np.array(context)
    options = np.array(options)
    np.save(loc.train_context_path, context)
    np.save(loc.train_options_path, options)
    print("Done.")

    # Create nparray of Valdation set
    print("Creating np_arrays of Validation set...")
    context, options = create_nparrays(input_filename=VALIDATION_PATH, vocab=vocab)
    print("Writing np_arrays of Validation set...")
    context = np.array(context)
    options = np.array(options)
    np.save(loc.valid_context_path, context)
    np.save(loc.valid_options_path, options)
    print("Done.")
    '''

    ### TODO: Make a function which writes followings to files
    # Create nparray of Training set
    '''
    print("Creating np_arrays of Training set...")
    context, options = create_nparrays2(input_filename=TRAIN_PATH, vocab=advising_vocab)
    print("Writing np_arrays of Training set...")
    context = np.array(context)
    options = np.array(options)
    np.save(loc.train_context_path, context)
    np.save(loc.train_options_path, options)
    print("Done.")
    '''
    # Create nparray of Valdation set
    print("Creating np_arrays of Validation set...")
    context, options = create_nparrays2(input_filename=VALIDATION_PATH, vocab=advising_vocab)
    print("Writing np_arrays of Validation set...")
    context = np.array(context)
    options = np.array(options)
    np.save(loc.valid_context_path, context)
    np.save(loc.valid_options_path, options)
    print("Done.")

