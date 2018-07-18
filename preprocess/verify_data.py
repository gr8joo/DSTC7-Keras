import os
import numpy as np

train_context = np.load('train_data/train_context.npy') # np.load('train_data/train_context_10.npy')#
train_context_len = np.load('train_data/train_context_len.npy')# np.load('train_data/train_context_len_10.npy')#
train_target = np.load('train_data/train_target.npy')# np.load('train_data/train_target_10.npy')#
train_options = np.load('train_data/train_options.npy')# np.load('train_data/train_options_10.npy')#
train_options_len = np.load('train_data/train_options_len.npy')# np.load('train_data/train_options_len_10.npy')#

valid_context = np.load('valid_data/valid_context.npy')# np.load('valid_data/valid_context_10.npy')#
valid_context_len = np.load('valid_data/valid_context_len.npy')# np.load('valid_data/valid_context_len_10.npy')#
valid_target = np.load('valid_data/valid_target.npy')# np.load('valid_data/valid_target_10.npy')#
valid_options = np.load('valid_data/valid_options.npy')# np.load('valid_data/valid_options_10.npy')#
valid_options_len = np.load('valid_data/valid_options_len.npy')# np.load('valid_data/valid_options_len_10.npy')#

print("====================================Train====================================")
print("context: ", train_context.shape)
print("context_len: ", train_context_len.shape)
print("target: ", train_target.shape)
print("options: ", train_options.shape)
print("options_len: ", train_options_len.shape)
print("")
print("====================================Valid====================================")
print("context: ", valid_context.shape)
print("context_len: ", valid_context_len.shape)
print("target: ", valid_target.shape)
print("options: ", valid_options.shape)
print("options_len: ", valid_options_len.shape)
print("")
print("=====================================ETC=====================================")
print(train_options[0][9])
print(valid_target)

a = set(train_target)
print("Train_target_set: ", a)
b = set(valid_target)
print("Valid_target_set: ", b)
print("=============================================================================")