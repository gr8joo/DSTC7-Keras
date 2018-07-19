import os
import sys
# sys.path.insert(0, '../')
# sys.path.append('../')

import ijson
import functools

import tensorflow as tf
import numpy as np

import stemmer
from locations import create_data_locations


tf.flags.DEFINE_integer(
    "min_word_frequency", 1, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string("train_in", None, "Path to input data file")
tf.flags.DEFINE_string("validation_in", None, "Path to validation data file")

tf.flags.DEFINE_string("train_out", None, "Path to output train tfrecords file")
tf.flags.DEFINE_string("validation_out", None, "Path to output validation tfrecords file")

tf.flags.DEFINE_string("vocab_path", None, "Path to save vocabulary txt file")
tf.flags.DEFINE_string("vocab_processor", None, "Path to save vocabulary processor")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.train_in)
VALIDATION_PATH = os.path.join(FLAGS.validation_in)

def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


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

        ###################### stemmer ######################
        new_context = stemmer.stem_sentence(context)
        new_next_utterances = []
        for utterance in next_utterances:
            new_utterance = stemmer.stem_sentence(utterance)
            new_next_utterances.append(new_utterance)

        context = new_context
        next_utterances = new_next_utterances
        #####################################################

        all_utterances.append(context)
        all_utterances.extend(next_utterances)
        for utterance in all_utterances:
            yield utterance

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


    ######################### stemmer ########################
    new_context = stemmer.stem_sentence(context)
    new_next_utterances = []
    for utterance in next_utterances:
        new_utterance = stemmer.stem_sentence(utterance)
        new_next_utterances.append(new_utterance)

    context = new_context
    next_utterances = new_next_utterances
    ##########################################################

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


if __name__ == "__main__":

    print("Creating vocabulary...")
    input_iter = create_dialog_iter(TRAIN_PATH)
    input_iter = create_utterance_iter(input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    write_vocabulary(
        vocab, os.path.join(FLAGS.vocab_path))

    # Save vocab processor
    vocab.save(os.path.join(FLAGS.vocab_processor))

    # Get directories of files to write
    loc = create_data_locations()
    
    ### TODO: Make a function which writes followings to files
    # Create nparray of Training set
    print("Creating np_arrays of Training set...")
    context, context_len, target, options, options_len = create_nparrays(
        input_filename=TRAIN_PATH,
        # output_filename=os.path.join(FLAGS.train_out),
        example_fn=functools.partial(create_example_nparray_format, vocab=vocab))
    print("Writing np_arrays of Training set...")
    context = np.array(context)
    context_len = np.array(context_len)
    target = np.array(target)
    options = np.array(options)
    options_len = np.array(options_len)

    np.save(loc.stem_train_context_path, context)
    np.save(loc.stem_train_context_len_path, context_len)
    np.save(loc.stem_train_target_path, target)
    np.save(loc.stem_train_options_path, options)
    np.save(loc.stem_train_options_len_path, options_len)
    print("Done.")

    # Create nparray of Valdation set
    print("Creating np_arrays of Validation set...")
    context, context_len, target, options, options_len = create_nparrays(
        input_filename=VALIDATION_PATH,
        # output_filename=os.path.join(FLAGS.validation_out),
        example_fn=functools.partial(create_example_nparray_format, vocab=vocab))
    print("Writing np_arrays of Validation set...")
    context = np.array(context)
    context_len = np.array(context_len)
    target = np.array(target)
    options = np.array(options)
    options_len = np.array(options_len)

    np.save(loc.stem_valid_context_path, context)
    np.save(loc.stem_valid_context_len_path, context_len)
    np.save(loc.stem_valid_target_path, target)
    np.save(loc.stem_valid_options_path, options)
    np.save(loc.stem_valid_options_len_path, options_len)
    print("Done.")
