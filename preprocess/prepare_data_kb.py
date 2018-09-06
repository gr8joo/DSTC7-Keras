import os
import sys
# sys.path.insert(0, '../')
# sys.path.append('../')

import ijson
import json
import linecache

import tensorflow as tf
import numpy as np

from locations import create_data_locations

tf.flags.DEFINE_integer(
    "min_word_frequency", 1, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string("train_in", None, "Path to input data file")
tf.flags.DEFINE_string("valid_in", None, "Path to validation data file")

tf.flags.DEFINE_string("train_out", None, "Path to output train tfrecords file")
tf.flags.DEFINE_string("valid_out", None, "Path to output validation tfrecords file")

tf.flags.DEFINE_string("vocab_path", None, "Path to save vocabulary txt file")
tf.flags.DEFINE_string("vocab_processor", None, "Path to save vocabulary processor")

FLAGS = tf.flags.FLAGS

# TRAIN_PATH = os.path.join(FLAGS.train_in)
VALIDATION_PATH = os.path.join(FLAGS.valid_in)


def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    :param dialog:
    :return:
    """

    row = []
    # row2= []
    utterances = dialog['messages-so-far']

    # Create the context
    context = ""
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context += "__eot__\n\n" + msg['utterance'] + " __eou__ "
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

    # Create profile of the student
    profile = dialog['profile']
    # categories = [key for val, key in enumerate(profile)]

    # print(len(categories[0]), len(categories[1]), len(categories[2]), len(categories[3]))
    return row, profile


if __name__ == "__main__":

    # Get directories of files to write
    loc = create_data_locations()


    target = np.load('valid_data/valid_advising/valid_target.npy')
    # target = np.load('train_data/train_advising/train_target.npy')
    with open(VALIDATION_PATH, 'rb') as f:
        json_data = json.load(f)
        while True:
    #idx = int(input("What number of example do you want to see? :  index starts from 0 : "))
            idx = int(input("Example index (starts from 0): "))
            row, profile = process_dialog(json_data[idx])
            print("============================ Context ============================")
            print(row[0], '\n')
            print("========================= Correct Answer ========================")
            print(str(target[idx])+': '+row[target[idx]+1]+'\n')
            print("============================ Profile ============================")

            print("Standing: "+str(profile['Standing']))
            print("\nTerm: "+str(profile['Term'])+'\n')

            Aggregated = profile['Aggregated']
            
            Area = [str(key)+': '+str(Aggregated['Area'][key]) for val, key in enumerate(Aggregated['Area'])]
            print("Area: ", Area)
            # Area = Aggregated['Area']

            ClarityRating = [str(key)+': '+str(Aggregated['ClarityRating'][key]) for val, key in enumerate(Aggregated['ClarityRating'])]
            print("\nClarityRating: ", ClarityRating)
            #Aggregated['ClarityRating']

            ClassSize = [str(key)+': '+str(Aggregated['ClassSize'][key]) for val, key in enumerate(Aggregated['ClassSize'])]
            print("\nClassSize: ", ClassSize)

            EasinessRating = [str(key)+': '+str(Aggregated['EasinessRating'][key]) for val, key in enumerate(Aggregated['EasinessRating'])]
            print("\nEasinessRating: ", EasinessRating)

            FractionGreaterThanEqualToA = [str(key)+': '+str(Aggregated['FractionGreaterThanEqualToA'][key]) for val, key in enumerate(Aggregated['FractionGreaterThanEqualToA'])]
            print("\nFractionGreaterThanEqualToA: ", FractionGreaterThanEqualToA)

            HelpfulnessRating = [str(key)+': '+str(Aggregated['HelpfulnessRating'][key]) for val, key in enumerate(Aggregated['HelpfulnessRating'])]
            print("\nHelpfulnessRating: ", HelpfulnessRating)

            TimeOfDay = [str(key)+': '+str(Aggregated['TimeOfDay'][key]) for val, key in enumerate(Aggregated['TimeOfDay'])]
            print("\nTimeOfDay: ", TimeOfDay)

            Workload = [str(key)+': '+str(Aggregated['Workload'][key]) for val, key in enumerate(Aggregated['Workload'])]
            print("\nWorkload: ", Workload)

            prior_courses = profile['Courses']['Prior']
            suggested_courses = profile['Courses']['Suggested']

            print("\nPrior courses:")
            for i in range(len(prior_courses)):
                print(prior_courses[i]['offering']+' / '+prior_courses[i]['instructor'])

            print("\nSuggested courses:")
            for i in range(len(suggested_courses)):
                print(suggested_courses[i]['offering']+' / '+suggested_courses[i]['instructor'])

            print('')
            print("============================ utterances ==========================")
            for i in range(100):
                print(str(i)+': '+row[i+1])
            print("\n")
            '''
            command = input("\nAnother one[y/n]? :")
            if command == 'n':
                break
            '''
