import os
import numpy as np
import random
from locations import create_data_locations

MAX_CONTEXT_LEN = 600#400
MAX_UTTERANCE_LEN = 140#90
loc = create_data_locations()

train_context = np.load(loc.train_context_path)
# train_context_len = np.load(loc.train_context_len_path)
# train_target = np.load(loc.train_target_path)
train_options = np.load(loc.train_options_path)
# train_options_len = np.load(loc.train_options_len_path)

valid_context = np.load(loc.valid_context_path)
# valid_context_len = np.load(loc.valid_context_len_path)
# valid_target = np.load(loc.valid_target_path)
valid_options = np.load(loc.valid_options_path)
# valid_options_len = np.load(loc.valid_options_len_path)


print("====================================Train====================================")
print("context: ", train_context.shape)
# print("context_len: ", train_context_len.shape)
# print("target: ", train_target.shape)
print("options: ", train_options.shape)
# print("options_len: ", train_options_len.shape)
print("")

print("====================================Valid====================================")
print("context: ", valid_context.shape)
# print("context_len: ", valid_context_len.shape)
# print("target: ", valid_target.shape)
print("options: ", valid_options.shape)
# print("options_len: ", valid_options_len.shape)
print("")
print("=====================================ETC=====================================")
# print(train_context[0])
print(valid_context[0])
# print(valid_target)

# a = set(train_target)
# print("Train_target_set: ", a)
# b = set(valid_target)
# print("Valid_target_set: ", b)
print("===================================Recovery======================================")
with open('/home/hjhwang/Codes/dstc7-keras/data/ubuntu_uKB_test_lemma_vocab.txt','r') as f:
    A = f.read()
    vocab = A.split('\n')

    print("Advising vocab exists!")
    '''
    # Visualize valid data (original)

    while True:
        idx = int(input("Example index (starts from 0): "))
        recovery_context = [ vocab[ valid_context[idx][i]-1 ]\
                                for i in range(MAX_CONTEXT_LEN) if valid_context[idx][i]>0]
        print("----------------------------Context----------------------------")
        str = ''
        for i in range(len(recovery_context)):
            # print(recovery_context[i])
            str += recovery_context[i] + ' '
            if recovery_context[i] == '__eou__':
                print(str)
                str = ''
        # do = [ int(valid_options[0][idx][i]) for i in range(100)]
        print("----------------------------Options----------------------------")
        for i in range(100):
            recovery = [ vocab[ valid_options[idx][i][j]-1 ]\
                                for j in range(MAX_UTTERANCE_LEN) if valid_options[idx][i][j]>0]
            str = ''
            for j in range(len(recovery)):
                # print(recovery_context[i])
                str += recovery[j] + ' '
            print(i,':', str)

    '''
    # Visualize training data (original)

    while True:
        idx = int(input("Example index (starts from 0): "))
        recovery_context = [ vocab[ train_context[idx][i]-1 ]\
                                for i in range(MAX_CONTEXT_LEN) if train_context[idx][i]>0]
        print("----------------------------Context----------------------------")
        str = ''
        for i in range(len(recovery_context)):
            # print(recovery_context[i])
            str += recovery_context[i] + ' '
            if recovery_context[i] == '__eou__':
                print(str)
                str = ''
        # do = [ int(valid_options[0][idx][i]) for i in range(100)]
        print("----------------------------Options----------------------------")
        for i in range(100):
            recovery = [ vocab[ train_options[idx][i][j]-1 ]\
                                for j in range(MAX_UTTERANCE_LEN) if train_options[idx][i][j]>0]
            str = ''
            for j in range(len(recovery)):
                # print(recovery_context[i])
                str += recovery[j] + ' '
            print(i,':', str)

    '''
    # Visualize train data (spitted context)
    while True:
        idx = int(input("Example index (starts from 0): "))

        print("----------------------------Context----------------------------")        
        for i in range(42):
            recovery_context = [ vocab[ train_context[idx][i][j]-1 ] for j in range(90) if train_context[idx][i][j]>0]
            str = ''
            for i in range(len(recovery_context)):
                # print(recovery_context[i])
                str += recovery_context[i] + ' '
            print(str)

        # do = [ int(valid_options[0][idx][i]) for i in range(100)]
        print("----------------------------Options----------------------------")
        for i in range(100):
            recovery = [ vocab[ train_options[idx][i][j]-1 ] for j in range(90) if train_options[idx][i][j]>0]
            str = ''
            for j in range(len(recovery)):
                # print(recovery_context[i])
                str += recovery[j] + ' '
            print(i,':', str)
    '''
    '''
    # Visualize validation data (spitted context)
    while True:
        idx = int(input("Example index (starts from 0): "))

        print("----------------------------Context----------------------------")        
        for i in range(42):
            recovery_context = [ vocab[ valid_context[idx][i][j]-1 ] for j in range(90) if valid_context[idx][i][j]>0]
            str = ''
            for i in range(len(recovery_context)):
                # print(recovery_context[i])
                str += recovery_context[i] + ' '
            print(str)

        # do = [ int(valid_options[0][idx][i]) for i in range(100)]
        print("----------------------------Options----------------------------")
        for i in range(100):
            recovery = [ vocab[ valid_options[idx][i][j]-1 ] for j in range(90) if valid_options[idx][i][j]>0]
            str = ''
            for j in range(len(recovery)):
                # print(recovery_context[i])
                str += recovery[j] + ' '
            print(i,':', str)
    '''
'''
indices = random.sample(range(100000), 20000)
train_X = [np.take(train_context, indices, axis=0), np.take(train_options, indices, axis=0)]
print("train_X[0]: ", train_X[0].shape)
print("train_X[1]: ", train_X[1].shape)
train_Y = np.take(train_target, indices, axis=0)
print("train_Y: ", train_Y.shape)
'''