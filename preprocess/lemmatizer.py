import os
import sys
import ijson
import functools
import numpy as np
#from locations import create_data_locations
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


part = {
    'N' : 'n',
    'V' : 'v',
    'J' : 'a',
    'S' : 's',
    'R' : 'r'
}

wnl = WordNetLemmatizer()

def tag_and_lem(element):
    '''
    tag_and_lem() accepts a string, tokenizes, tags, converts tags,
    lemmatizes, and returns a string
    '''
    # list of tuples [('token', 'tag'), ('token2', 'tag2')...]
    sent = pos_tag(word_tokenize(element)) # must tag in context
    return ' '.join([wnl.lemmatize(sent[k][0], convert_tag(sent[k][1][0]))
                    for k in range(len(sent))])

def convert_tag(penn_tag):
    '''
    convert_tag() accepts the **first letter** of a Penn part-of-speech tag,
    then uses a dict lookup to convert it to the appropriate WordNet tag.
    '''
    if penn_tag in part.keys():
        return part[penn_tag]
    else:
        # other parts of speech will be tagged as nouns
        return 'n'


def stem_sentence(sentence):

    """
    
    Takes STRING as an input, returns a STEMMED STRING
    
    """

#    ps = nltk.stem.PorterStemmer()
#    ls = LancasterStemmer()
#    lemmatizer = WordNetLemmatizer()

    #print(sentence) # For the purpose of debugging, print original sentence
    sentence = sentence.strip().replace('\n',' ').replace('..',' ')
    new_sentence = tag_and_lem(sentence.lower()).replace('eec ','eecs ').replace(' c ',' cs ')
    
    #print(new_sentence)
    
    return new_sentence
    
    
    
    
    
    
""" 
    stemmed_sentence = []
    sentence = sentence.replace('\n',' ')
    sentence = sentence.replace('.',' .')
    sentence = sentence.replace(',',' ,')
    sentence = sentence.replace('!',' !')
    sentence = sentence.replace('?',' ?')
#    print(sentence)
    
    for word in sentence.strip().split(" "):
#        print(word)
        #stemmed_sentence.append(ps.stem(word))
        #stemmed_sentence.append(ls.stem(word))
        word = word.lower()
        
        #if word in ['has','was','includes']:
        #    new_word = lemmatizer.lemmatize(word,pos='v')
        #else if word in ['eecs','cs']:
        #    new_word = word
        #else:
        #    new_word = lemmatizer.lemmatize(word)
        new_word = tag_and_lem(word)
        if new_word == 'eec':
            new_word = 'eecs'
#        print(new_word)

        stemmed_sentence.append(new_word)
    new_sentence = " ".join(stemmed_sentence)
    
    print(new_sentence) # For the purpose of debugging, print the converted sentence
    
    return new_sentence
"""   
def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    :param dialog:
    :return:
    """

    row = []
    utterances = dialog['messages-so-far']

    # Create the context
    context = ""
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context += "__eot__ " + msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        else:
            context += msg['utterance'] + " __eou__ "

    context += "__eot__"
    row.append(context)

    # Create the next utterance options and the target label
    correct_answer = dialog['options-for-correct-answers'][0]
    target_id = correct_answer['candidate-id']
    target_index = None
    for i, utterance in enumerate(dialog['options-for-next']):
        if utterance['candidate-id'] == target_id:
            target_index = i
        row.append(utterance['utterance'] + " __eou__ ")

    if target_index is None:
        print('Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(dialog['example-id']))
    else:
        row.append(target_index)

    return row


def create_dialog_iter(filename):
    """
    Returns an iterator over a JSON file.
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            row = process_dialog(entry)
            yield row

def create_utterance_iter(input_iter):
    """
    Returns an iterator over every utterance (context and candidates) for the VocabularyProcessor.
    :param input_iter:
    :return:
    """
    for row in input_iter:
        all_utterances = []
        context = row[0]
        next_utterances = row[1:101]
#
        ###################### stemmer ######################
#        print(context)
#        print(next_utterances)
#        new_context = stem_sentence(context)
#        new_next_utterances = []
#        for utterance in next_utterances:
#            new_utterance = stem_sentence(utterance)
#            new_next_utterances.append(new_utterance)
#
#        context = new_context
#        next_utterances = new_next_utterances
        #####################################################
#
        all_utterances.append(context)
        all_utterances.extend(next_utterances)
        
        for utterance in all_utterances:
            #print(utterance)
            new_utterance = stem_sentence(utterance)
            #new_utterance =tag_and_lem(utterance)
            #print(new_utterance)
            yield new_utterance

def create_vocab(input_iter, min_frequency):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary
    for the input iterator.
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)
    return vocab_processor


def transform_sentence(sequence, vocab_processor):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array.
    """
    return next(vocab_processor.transform([sequence])).tolist()


def create_example_nparray_format(row, vocab):

    context = row[0]
    next_utterances = row[1:101]
#
#
    ######################### stemmer ########################
    new_context = stem_sentence(context)
    new_next_utterances = []
    for utterance in next_utterances:
        new_utterance = stem_sentence(utterance)
        new_next_utterances.append(new_utterance)#
#
    context = new_context
    next_utterances = new_next_utterances
    ##########################################################
#
    target = row[-1]

    context_transformed = transform_sentence(context, vocab)
    context_len = len(next(vocab._tokenizer([context])))

    # Distractor sequences
    options = []
    options_len = []
    for i, utterance in enumerate(next_utterances):
        opt_key = "option_{}".format(i)
        opt_len_key = "option_{}_len".format(i)
        # Utterance Length Feature
        opt_len = len(next(vocab._tokenizer([utterance])))
        
        # Distractor Text Feature
        opt_transformed = transform_sentence(utterance, vocab)
        
        options.append(opt_transformed)
        options_len.append(opt_len)
    return context_transformed, context_len, target, options, options_len


def create_nparrays(input_filename, example_fn):
    """
    Creates nparrays for the given input data and
    example transofmration function
    """
    context_chunk=[]
    context_len_chunk=[]
    target_chunk=[]
    options_chunk=[]
    options_len_chunk=[]
    for i, row in enumerate(create_dialog_iter(input_filename)):
        context, context_len, target, options, options_len = example_fn(row)
        context_chunk.append(context)
        context_len_chunk.append(context_len)
        target_chunk.append(target)
        options_chunk.append(options)
        options_len_chunk.append(options_len)

    return context_chunk, context_len_chunk, target_chunk, options_chunk, options_len_chunk


def write_vocabulary(vocab_processor, outfile):
    """
    Writes the vocabulary to a file, one word per line.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, "w") as vocabfile:
        for id in range(vocab_size):
            word =  vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("Saved vocabulary to {}".format(outfile))

#input_iter1 = create_dialog_iter('advising.scenario-1.train.json')
#input_iter2 = create_dialog_iter('advising.scenario-1.dev.json')
#input_iter1 = create_utterance_iter(input_iter1)
#input_iter2 = create_utterance_iter(input_iter2)
input1 = open('wiki.en.txt','r')
#input2 = open('advising_train_and_dev_text.txt','r')


#f = open("advising_train_and_dev_lemmatized_final_text.txt",'w')
#fout1 = open("advising_train_lemmatized_final2_text.txt",'w')
#fout2 = open("advising_dev_lemmatized_final2_text.txt",'w')
#f = open("wiki_advising_merged_lemmatized_final_text.txt",'w')
f = open("wiki_lemmatized_final_text.txt",'w')

i=0
#for sentence in input_iter1:
"""
print("Start writing from advising_train_and_dev_merged_text.txt...")

for sentence in input2:
    #sentence.replace('\n',' ')
    #fout1.write(sentence+'\n')
    new_sentence = stem_sentence(sentence)
    f.write(new_sentence+'\n')
    i = i+1
    if i % 100000 == 0:
        print(i)
#for sentence in input_iter2:
"""

print("Start writing from wiki.en.txt...")

for sentence in input1:
    #sentence.replace('\n',' ')
    #fout2.write(sentence+'\n')
    new_sentence = stem_sentence(sentence)
    f.write(new_sentence+'\n')
    i = i+1
    if i % 100000 == 0:
        print(i)
print(i)
#fout1.close()
#fout2.close()
input1.close()
#input2.close()
f.close()

#10150500
