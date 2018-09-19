import os
import sys
# sys.path.insert(0, '../')
# sys.path.append('../')

import ijson
import json
import functools
import pickle
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import normalize
from locations import create_data_locations

tf.flags.DEFINE_integer(
    "min_word_frequency", 1, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_string("train_lemma_in", None, "Path to training data file")
tf.flags.DEFINE_string("train_json_in", None, "Path to training data file")

tf.flags.DEFINE_string("valid_lemma_in", None, "Path to validation data file")
tf.flags.DEFINE_string("valid_json_in", None, "Path to validation data file")

tf.flags.DEFINE_string("test_lemma_in", None, "Path to testing data file")
tf.flags.DEFINE_string("test_json_in", None, "Path to testing data file")


tf.flags.DEFINE_string("vocab", None, "Path to save vocabulary processor")
tf.flags.DEFINE_string("entire_vocab", None, "Path to save vocabulary txt file")

tf.flags.DEFINE_string("embedding_path", None, "Path to save vocabulary txt file")

FLAGS = tf.flags.FLAGS

TRAIN_LEMMA_PATH= os.path.join(FLAGS.train_lemma_in)
TRAIN_JSON_PATH = os.path.join(FLAGS.train_json_in)

# VALID_LEMMA_PATH= os.path.join(FLAGS.valid_lemma_in)
# VALID_JSON_PATH = os.path.join(FLAGS.valid_json_in)

# TEST_LEMMA_PATH= os.path.join(FLAGS.test_lemma_in)
# TEST_JSON_PATH = os.path.join(FLAGS.test_json_in)

TRAIN_LABEL = 'train_data/train_ubuntu/train_target.npy'
VALID_LABEL = 'valid_data/valid_ubuntu/valid_target.npy'


# VOCAB_PATH = os.path.join(FLAGS.vocab_path)
# ENTIRE_VOCAB_PATH = os.path.join(FLAGS.entire_vocab_path)

MAX_CONTEXT_LEN = 600#1250#400 for Advising
MAX_UTTERANCE_LEN = 140#230#90 for Advising

MAX_SUGGESTED_COURSES_LEN = 24  # for Advising only
MAX_UTTR_NUM = 42               # Deprecated

def zerolistmaker(n):
    listofzeros= [0] * n
    return listofzeros

def onelistmaker(n):
    listofones = [1] * n
    return listofones

def count_len(input_filename):
    max_context_len = 0
    max_utterance_len = 0

    context_len_list = zerolistmaker(20)
    utterance_len_list = zerolistmaker(20)

    max_uttr_num = 0
    max_uttr_len = 0
    with open(input_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line: 
                break
            # line = line.replace('\n', ' ')
            line = line[:-1]
            word = line.split()

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
                word = line.split()
                utterance_len = len(word)
                utterance_len_list[ int(utterance_len/100)] += 1
                if utterance_len > max_utterance_len:
                    max_utterance_len = utterance_len
                ### if utterance_len == 89:
                ###     print(word)
    return context_len_list, utterance_len_list, max_context_len, max_utterance_len, max_uttr_num, max_uttr_len

def count_true_len(input_filename, input_labelname):
    max_context_len = 0
    max_utterance_len = 0
    max_true_utterance_len = 0
    context_len_list = zerolistmaker(20)
    utterance_len_list = zerolistmaker(20)
    true_utterance_len_list = zerolistmaker(20)
    idx = 0
    label = np.load(input_labelname)
    with open(input_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line: 
                break
            # line = line.replace('\n', ' ')
            line = line[:-1]

            word = line.split()
            context_len = len(word)
            context_len_list[ int(context_len/100)] += 1
            if context_len > max_context_len:
                max_context_len = context_len

            for i in range(100):
                line = f.readline()
                # line = line.replace('\n', ' ')
                line = line[:-1]
                word = line.split()
                utterance_len = len(word)
                utterance_len_list[ int(utterance_len/100)] += 1
                if utterance_len > max_utterance_len:
                    max_utterance_len = utterance_len

                if i == label[idx]:
                    true_utterance_len = len(word)
                    true_utterance_len_list[ int(utterance_len/100)] += 1
                    if true_utterance_len > max_true_utterance_len:
                        max_true_utterance_len = true_utterance_len
                ### if utterance_len == 89:
                ###     print(word)

            idx=idx+1
    return context_len_list, utterance_len_list, true_utterance_len_list,\
            max_context_len, max_utterance_len, max_true_utterance_len

def create_nparrays(input_filename, loc, mode, vocab):

    # json_data=[]
    # with open(json_filename, 'rb') as g:
    #    json_data = json.load(g)
    if mode == 1:
        CONTEXT_PATH = loc.train_context_path
        CONTEXT_MASK_PATH = loc.train_context_mask_path
        CONTEXT_LEN_PATH = loc.train_context_len_path

        OPTIONS_PATH = loc.train_options_path
        OPTIONS_LEN_PATH = loc.train_options_len_path

    else:
        CONTEXT_PATH = loc.valid_context_path# 'test1.file'#
        CONTEXT_MASK_PATH = loc.valid_context_mask_path
        CONTEXT_LEN_PATH = loc.valid_context_len_path

        OPTIONS_PATH = loc.valid_options_path# 'test3.file'#
        OPTIONS_LEN_PATH = loc.valid_options_len_path


    with open(input_filename, 'r') as f:
        
        context_set = []
        context_mask_set = []
        context_len_set = []

        utterance_set = []
        utterance_len_set = []
        
        idx = 0
        while True:
            # Handle context
            line = f.readline()
            if not line: 
                break

            # eliminating '\n'
            line = line[:-1]
            line = line.split()

            # elimianating eot
            context = line
            context = [x for x in context if x != '__eot__']
            context_len = len(context)
    
            context_transformed = [vocab[context[i]]+1 for i in range(context_len)]
            context_mask = []
            if context_len < MAX_CONTEXT_LEN:
                context_transformed.extend(zerolistmaker(MAX_CONTEXT_LEN-context_len))
                context_mask = zerolistmaker(context_len)
                context_mask.extend(onelistmaker(MAX_CONTEXT_LEN-context_len))
            
            elif context_len >= MAX_CONTEXT_LEN:
                context_transformed = context_transformed[context_len-MAX_CONTEXT_LEN:context_len]
                context_mask = zerolistmaker(MAX_CONTEXT_LEN)

                context_len = MAX_CONTEXT_LEN
            
            context_set.append(context_transformed)
            context_mask_set.append(context_mask)
            context_len_set.append(context_len)

            
            temp_utterance_set = []
            temp_utterance_len_set = []
            for i in range(100):
                line = f.readline()
                
                # eliminating '\n'
                line = line[:-1]
                utterance = line.split()
                utterance_len = len(utterance)

                utterance_transformed = [vocab[utterance[i]]+1 for i in range(utterance_len)]
                if utterance_len < MAX_UTTERANCE_LEN:
                    utterance_transformed.extend(zerolistmaker(MAX_UTTERANCE_LEN-utterance_len))
                                
                elif utterance_len >= MAX_UTTERANCE_LEN:
                    utterance_transformed = utterance_transformed[:MAX_UTTERANCE_LEN]
                    utterance_len = MAX_UTTERANCE_LEN
                                
                temp_utterance_set.append(utterance_transformed)
                temp_utterance_len_set.append(utterance_len)
                
            utterance_set.append(temp_utterance_set)
            utterance_len_set.append(temp_utterance_len_set)
            # np.save(o, np.array(temp_utterance_set))
            # pickle.dump(np.array(temp_utterance_set), o)
            
            idx = idx+1
            if idx%100 == 0:
                print(idx, 'samples')


    context_set = np.array(context_set, dtype='i4')
    context_mask_set = np.array(context_mask_set, dtype='i4')
    context_len_set = np.array(context_len_set, dtype='i4')

    # import pdb; pdb.set_trace()
    utterance_set = np.array(utterance_set, dtype='i4')
    utterance_len_set = np.array(utterance_len_set, dtype='i4')

    np.save(CONTEXT_PATH, context_set)
    np.save(CONTEXT_MASK_PATH, context_mask_set)
    np.save(CONTEXT_LEN_PATH, context_len_set)

    np.save(OPTIONS_PATH, utterance_set)
    np.save(OPTIONS_LEN_PATH, utterance_len_set)
    
    # return context_set, context_speaker_set, utterance_set, utterance_speaker_set

def create_nparrays_with_speaker(input_filename, json_filename, loc, mode, vocab):

    # json_data=[]
    # with open(json_filename, 'rb') as g:
    #    json_data = json.load(g)
    if mode == 1:
        CONTEXT_PATH = loc.train_context_path
        CONTEXT_SPEAKER_PATH = loc.train_context_speaker_path
        CONTEXT_MASK_PATH = loc.train_context_mask_path
        CONTEXT_LEN_PATH = loc.train_context_len_path

        OPTIONS_PATH = loc.train_options_path
        OPTIONS_LEN_PATH = loc.train_options_len_path

    else:
        CONTEXT_PATH = loc.valid_context_path# 'test1.file'#
        CONTEXT_SPEAKER_PATH = loc.valid_context_speaker_path# 'test2.file'#
        CONTEXT_MASK_PATH = loc.valid_context_mask_path
        CONTEXT_LEN_PATH = loc.valid_context_len_path

        OPTIONS_PATH = loc.valid_options_path# 'test3.file'#
        OPTIONS_LEN_PATH = loc.valid_options_len_path


    with open(input_filename, 'r') as f, open(json_filename, 'rb') as g:
        
        context_set = []
        context_speaker_set = []
        context_mask_set = []
        context_len_set = []

        utterance_set = []
        utterance_len_set = []
        
        idx = 0
        for json_data in ijson.items(g, "item"):
            # Handle context
            line = f.readline()
            if not line: 
                break

            # eliminating '\n'
            line = line[:-1]
            line = line.split()

            # elimianating eot
            context = []
            context_speaker = []
            
            speaker_info = [1, 0]
            if json_data['messages-so-far'][0]['speaker'] == 'student':
                speaker_info = [0, 1]
            
            for item in line:
                if item != '__eot__':
                    context.append(item)
                    context_speaker.append(speaker_info)
                else:
                    if speaker_info[0] == 0:
                        speaker_info = [1, 0]
                    else:
                        speaker_info = [0, 1]

                    # speaker_info[0], speaker_info[1] = speaker_info[1], speaker_info[0]
                    # print(speaker_info)
            context_len = len(context)
    
            context_transformed = [vocab[context[i]]+1 for i in range(context_len)]
            context_mask = []
            if context_len < MAX_CONTEXT_LEN:
                context_transformed.extend(zerolistmaker(MAX_CONTEXT_LEN-context_len))
                context_speaker.extend([[0,0] for k in range(MAX_CONTEXT_LEN-context_len)])
                context_mask = zerolistmaker(context_len)
                context_mask.extend(onelistmaker(MAX_CONTEXT_LEN-context_len))
            
            elif context_len >= MAX_CONTEXT_LEN:
                context_transformed = context_transformed[context_len-MAX_CONTEXT_LEN:context_len]
                context_speaker = context_speaker[context_len-MAX_CONTEXT_LEN:context_len]
                context_mask = zerolistmaker(MAX_CONTEXT_LEN)

                context_len = MAX_CONTEXT_LEN
            
            context_set.append(context_transformed)
            context_speaker_set.append(context_speaker)
            context_mask_set.append(context_mask)
            context_len_set.append(context_len)

            
            temp_utterance_set = []
            temp_utterance_len_set = []
            for i in range(100):
                line = f.readline()
                
                # eliminating '\n'
                line = line[:-1]
                utterance = line.split()
                utterance_len = len(utterance)

                utterance_transformed = [vocab[utterance[i]]+1 for i in range(utterance_len)]
                if utterance_len < MAX_UTTERANCE_LEN:
                    utterance_transformed.extend(zerolistmaker(MAX_UTTERANCE_LEN-utterance_len))
                                
                elif utterance_len >= MAX_UTTERANCE_LEN:
                    utterance_transformed = utterance_transformed[:MAX_UTTERANCE_LEN]
                    utterance_len = MAX_UTTERANCE_LEN
                                
                temp_utterance_set.append(utterance_transformed)
                temp_utterance_len_set.append(utterance_len)
                
            utterance_set.append(temp_utterance_set)
            utterance_len_set.append(temp_utterance_len_set)
            # np.save(o, np.array(temp_utterance_set))
            # pickle.dump(np.array(temp_utterance_set), o)
            
            idx = idx+1
            if idx%1000 == 0:
                print(idx/1000, 'percent')
                # break
                # print(idx, 'samples')


    context_set = np.array(context_set, dtype='i4')
    context_speaker_set = np.array(context_speaker_set, dtype='i4')
    context_mask_set = np.array(context_mask_set, dtype='i4')
    context_len_set = np.array(context_len_set, dtype='i4')

    # import pdb; pdb.set_trace()
    utterance_set = np.array(utterance_set, dtype='i4')
    utterance_len_set = np.array(utterance_len_set, dtype='i4')

    np.save(CONTEXT_PATH, context_set)
    np.save(CONTEXT_SPEAKER_PATH, context_speaker_set)
    np.save(CONTEXT_MASK_PATH, context_mask_set)
    np.save(CONTEXT_LEN_PATH, context_len_set)

    np.save(OPTIONS_PATH, utterance_set)
    np.save(OPTIONS_LEN_PATH, utterance_len_set)
    
    # return context_set, context_speaker_set, utterance_set, utterance_speaker_set


def create_nparrays_split_context(input_filename, vocab):

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
            context = line.split()
            
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
                utterance = line.split()
                
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

def create_profile(json_filename, loc, mode, vocab):

    if mode == 1:
        PROFILE_PATH = loc.train_profile_path
    else:
        PROFILE_PATH = loc.valid_profile_path

    max_suggested_courses_len= 0
    suggested_courses_set = []
    with open(json_filename, 'rb') as g:
        
        
        idx = 0

        for json_data in ijson.items(g, "item"):
            suggested_courses_raw = json_data['profile']['Courses']['Suggested']
            suggested_courses = []
            suggested_courses_len = 0
            ### suggested_courses_len = len(suggested_courses_raw)
            ### if max_suggested_courses_len < suggested_courses_len:
            ###     max_suggested_courses_len = suggested_courses_len

            for data in suggested_courses_raw:
                course_name = data['offering'][:data['offering'].index('-')].lower()
                if course_name in vocab:
                    suggested_courses.append(vocab.index(course_name))
                    suggested_courses_len += 1
            if max_suggested_courses_len < suggested_courses_len:
                max_suggested_courses_len = suggested_courses_len

            if suggested_courses_len < MAX_SUGGESTED_COURSES_LEN:
                suggested_courses.extend(zerolistmaker(MAX_SUGGESTED_COURSES_LEN-suggested_courses_len))

            
            elif suggested_courses_len > MAX_SUGGESTED_COURSES_LEN:
                print('There must be something wrong!!!')
                break
            
            suggested_courses_set.append(suggested_courses)

            idx = idx+1
            if idx%10000 == 0:
                print(idx/1000, 'percent')


    suggested_courses_set = np.array(suggested_courses_set, dtype='i4')
    np.save(PROFILE_PATH, suggested_courses_set)


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
            word = line.split()
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
    with open(vocab_path, 'w') as thefile:
        for item in vocab:
            thefile.write("%s\n" % item)

    # embedding_W = np.array(embedding_W)
    # np.save('/home/hjhwang/Codes/dstc7-keras/data/embedding_W.npy', embedding_W)
    
    return vocab, vocab_size

if __name__ == "__main__":

    # Get directories of files to write
    loc = create_data_locations()

    ####### Create advising vocab #######
    vocab = []
    vocab_dict = {}
    vocab_size = 0
    if FLAGS.vocab is not None:
        if os.path.isfile(FLAGS.vocab) is False:
            print("Vocab does not exist... Creating to:", FLAGS.vocab)
            vocab, vocab_size =\
            generate_vocab_from_gensim('/home/hjhwang/word2vec/ubuntu_uKB_test_lemmatized.vector',
                                        FLAGS.vocab)
        else:
            print("Data Vocab exists!")
            f = open(FLAGS.vocab,'r')

            A = f.read()
            vocab = A.split('\n')
            vocab_size = len(A)
            
            for i,line in enumerate(vocab):
                vocab_dict[line]=i


    ####### Create entire vocab #######
    entire_vocab_dict = {}
    entire_vocab_size = 0
    if FLAGS.entire_vocab is not None:
        if os.path.isfile(FLAGS.entire_vocab) is False:
            print("Entire Vocab does not exist... Creating to", FLAGS.entire_vocab)
            entire_vocab, entire_vocab_size =\
            generate_vocab_from_gensim('/home/hjhwang/word2vec/wiki_ubuntu_uKB_test_lemmatized.vector',
                                        FLAGS.entire_vocab)
        else:
            print("Entire Vocab exists!")
            f = open(FLAGS.entire_vocab,'r')
            '''
            A = f.read()
            entire_vocab = A.split('\n')
            entire_vocab_size = len(A)
            '''
            for i,line in enumerate(f):
                entire_vocab_dict[line.strip()]=i


    ####### Extract advising embedding matrix from entire embedding matrix #######
    if FLAGS.embedding_path is not None:
        ### Trick ###
        # entire_vocab = vocab        # You sure???


        print("Creating embedding matirx to", FLAGS.embedding_path)
        # entire_embedding_W = np.load('/home/hjhwang/word2vec/wiki_advising_lemmatized.model.trainables.syn1neg.npy')
        entire_embedding_W = np.load('/home/hjhwang/word2vec/ubuntu/wiki_ubuntu_uKB_test_lemmatized/wiki_ubuntu_uKB_test_lemmatized.model.trainables.syn1neg.npy')
        print("Entire embedding loaded!")
        embedding_W = [np.zeros(300,)]
        cnt = 0
        for entry in vocab:
            if entry == '':
                break
            elif entry =='cyptography\n':
                print('Hack!!')
            idx = entire_vocab_dict[entry]
            # A = entire_embedding_W[idx]/np.linalg.norm(entire_embedding_W[idx])
            x = entire_embedding_W[idx]
            A = normalize(x[:,np.newaxis], axis=0).ravel()
            embedding_W.append(A)

        embedding_W = np.array(embedding_W)
        print("shape of embedding_W: ", embedding_W.shape)
        # np.save('/home/hjhwang/Codes/dstc7-keras/data/embedding_W.npy', embedding_W)
        np.save(FLAGS.embedding_path, embedding_W)

    ####### Count len #######

    '''
    print("In Training dataset: (Context / Utterance)")
    context_len_list, utterance_len_list,\
    max_context_len, max_utterance_len,\
    max_uttr_num, max_uttr_len = count_len(TRAIN_LEMMA_PATH)
    for i in range(20):
        print("len less than ",(i+1)*100, ': ', context_len_list[i], ' / ', utterance_len_list[i])
    print("max len: ", max_context_len, ' / ', max_utterance_len)
    print("max number of uttr in Context:", max_uttr_num)
    print("max len of uttr in Context:", max_uttr_len+1)
    print()

    print("In Validation dataset: (Context / Utterance)")
    context_len_list, utterance_len_list,\
    max_context_len, max_utterance_len,\
    max_uttr_num, max_uttr_len = count_len(VALID_LEMMA_PATH)
    for i in range(20):
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
    context, options = create_nparrays_split_context(input_filename=TRAIN_PATH, vocab=vocab)
    print("Writing np_arrays of Training set...")
    context = np.array(context)
    options = np.array(options)
    np.save(loc.train_context_path, context)
    np.save(loc.train_options_path, options)
    print("Done.")
    
    # Create nparray of Valdation set
    print("Creating np_arrays of Validation set...")
    context, options = create_nparrays_split_context(input_filename=VALIDATION_PATH, vocab=vocab)
    print("Writing np_arrays of Validation set...")
    context = np.array(context)
    options = np.array(options)
    np.save(loc.valid_context_path, context)
    np.save(loc.valid_options_path, options)
    print("Done.")
    '''

    '''
    print("Creating np_arrays of Training set...")
    context, context_speaker, options, options_speaker\
    = create_nparrays_with_speaker(input_filename=TRAIN_LEMMA_PATH, json_filename=TRAIN_JSON_PATH, loc=loc, mode=1, vocab=vocab)
    print("Writing np_arrays of TRAINING set...")
    context = np.array(context)
    context_speaker = np.array(context_speaker)
    options = np.array(options)
    options_speaker = np.array(options_speaker)
    np.save(loc.train_context_path, context)
    np.save(loc.train_context_speaker_path, context_speaker)
    np.save(loc.train_options_path, options)
    np.save(loc.train_options_speaker_path, options_speaker)
    print("Done.")


    print("Creating np_arrays of Validation set...")
    context, context_speaker, options, options_speaker\
    = create_nparrays_with_speaker(input_filename=VALID_LEMMA_PATH, json_filename=VALID_JSON_PATH, loc=loc, mode=2, vocab=vocab)
    print("Writing np_arrays of Validation set...")
    context = np.array(context)
    context_speaker = np.array(context_speaker)
    options = np.array(options)
    options_speaker = np.array(options_speaker)
    np.save(loc.valid_context_path, context)
    np.save(loc.valid_context_speaker_path, context_speaker)
    np.save(loc.valid_options_path, options)
    np.save(loc.valid_options_speaker_path, options_speaker)
    print("Done.")
    '''

    ##########################################################
    '''
    print("In Training dataset: (Context / Utterance / True Utterance)")
    context_len_list, utterance_len_list, true_utterance_len_list,\
    max_context_len, max_utterance_len, max_true_utterance_len\
    = count_true_len(TRAIN_LEMMA_PATH, TRAIN_LABEL)
    for i in range(20):
        print("len less than ",(i+1)*100, ': ', context_len_list[i], ' / ',
                                                utterance_len_list[i], ' / ',
                                                true_utterance_len_list[i])
    print("max len: ", max_context_len, ' / ', max_utterance_len, ' / ', max_true_utterance_len)

    print("In Validation dataset: (Context / Utterance / True Utterance)")
    context_len_list, utterance_len_list, true_utterance_len_list,\
    max_context_len, max_utterance_len, max_true_utterance_len\
    = count_true_len(VALID_LEMMA_PATH, VALID_LABEL)
    for i in range(20):
        print("len less than ",(i+1)*100, ': ', context_len_list[i], ' / ',
                                                utterance_len_list[i], ' / ',
                                                true_utterance_len_list[i])
    print("max len: ", max_context_len, ' / ',  max_utterance_len, ' / ', max_true_utterance_len)
    '''



    print("Creating np_arrays of Training set...")
    create_nparrays_with_speaker(input_filename=TRAIN_LEMMA_PATH,
                                    json_filename=TRAIN_JSON_PATH,
                                    loc=loc,
                                    mode=1,
                                    vocab=vocab_dict)
    print("Done.")
    '''
    print("Creating np_arrays of Validation set...")
    create_nparrays_with_speaker(input_filename=VALID_LEMMA_PATH,
                                    json_filename=VALID_JSON_PATH,
                                    loc=loc,
                                    mode=2,
                                    vocab=vocab_dict)
    print("Done.")


    print("Creating np_arrays of Testing set...")
    create_nparrays_with_speaker(input_filename=TEST_LEMMA_PATH,
                                    json_filename=TEST_JSON_PATH,
                                    loc=loc,
                                    mode=3,
                                    vocab=vocab)
    print("Done.")
    '''

    '''
    count = create_profile(json_filename=TRAIN_JSON_PATH, loc=loc, mode=1, vocab=vocab)
    print(count)
    count = create_profile(json_filename=VALID_JSON_PATH, loc=loc, mode=2, vocab=vocab)
    print(count)
    '''