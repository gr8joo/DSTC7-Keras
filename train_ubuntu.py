import os
import time
import numpy as np
import keras
import random
import pickle

from keras import backend as K
from keras import metrics
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
import models.helpers as helpers

from hyperparams import create_hyper_parameters

from models.memLstm2_ubuntu import memLstm_custom_model

from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint


def hack_loss(y_true, y_pred):
        return K.zeros((1,))

def top2acc(y_true, y_pred, k=2):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top5acc(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top10acc(y_true, y_pred, k=10):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top50acc(y_true, y_pred, k=50):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def main():

    ############################# Load Configurations #############################    
    hparams = create_hyper_parameters()


    ############### Load a model and shaping model input&output ###############
    context = Input(shape=(hparams.max_context_len,))
    # context_speaker = Input(shape=(hparams.max_context_len, 2))
    context_mask = Input(shape=(hparams.max_context_len,))
    utterances = Input(shape=(hparams.num_utterance_options, hparams.max_utterance_len))
    # profile = Input(shape=(hparams.max_profile_len,))


    inputs = [context, context_mask, utterances]
    # inputs = [context, context_speaker, utterances, profile]
    

    probs,\
    context_attention,\
    responses_attention,\
    responses_dot = memLstm_custom_model(hparams, context, context_mask, utterances)
    # probs = memLstm_model(hparams, context, context_speaker, utterances, profile)


    model = Model(inputs=inputs, outputs=[probs,
                                            context_attention,
                                            responses_attention,
                                            responses_dot])
    print("Model loaded")


    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    tensorboard =\
    TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # checkpointer =\
    # ModelCheckpoint(filepath='./dual_encoder_checkpoint.h5', verbose=1, save_best_only=True)
    # optim = keras.optimizers.SGD(lr=hparams.learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    optim = keras.optimizers.Adam(lr=hparams.learning_rate, clipnorm=hparams.clip_norm)#, decay=0.001)
    model.compile(loss={'probs': 'sparse_categorical_crossentropy',
                        'context_attention': hack_loss,
                        'responses_attention': hack_loss,
                        'responses_dot': hack_loss},#custom_loss(probs=probs),#{'probs': custom_loss},
                    optimizer=optim,
                    loss_weights={'probs': 1.0,
                                    'context_attention': 0.0,
                                    'responses_attention': 0.0,
                                    'responses_dot': 0.0},
                    metrics=['accuracy'])#, top2acc, top5acc, top10acc, top50acc])
    model.summary()


    ############################# Load Validation Datas #############################    
    print("Loading validation data")
    valid_context = np.load(hparams.valid_context_path)
    # valid_context_speaker =np.load(hparams.valid_context_speaker_path)
    valid_context_mask = np.load(hparams.valid_context_mask_path)
    # valid_context_len = np.load(hparam.valid_context_len_path)
    valid_target = np.load(hparams.valid_target_path)
    valid_target = valid_target.astype('i4')
    valid_options = np.load(hparams.valid_options_path)
    # valid_options_len = np.load(hparam.valid_context_path)
    # valid_profile = np.load(hparams.valid_profile_path)


    valid_context_mask = hparams.neg_inf * valid_context_mask

    
    valid_X = [valid_context, valid_context_mask, valid_options]
    # valid_X = [valid_context, valid_context_speaker, valid_options, valid_profile]
    valid_Y = [valid_target, np.zeros((5000,1,1),dtype='i4'),
                                np.zeros((5000,1,1,1,1), dtype='i4'),
                                np.zeros((5000,1,1,1,1), dtype='i4')]


    ############################# TRAIN #############################
    print("Loading training data")
    train_context = np.load(hparams.train_context_path)
    # train_context_speaker =np.load(hparams.train_context_speaker_path)
    train_context_mask = np.load(hparams.train_context_mask_path)
    # train_context_len = np.load(hparam.train_context_len_path)
    train_target = np.load(hparams.train_target_path)
    train_target = train_target.astype('i4')
    train_options = np.load(hparams.train_options_path)
    # train_options_len = np.load(hparam.train_context_path)
    # train_profile = np.load(hparams.train_profile_path)


    train_context_mask = hparams.neg_inf * train_context_mask


    # train_X = [train_context, train_context_speaker, train_options]
    # train_X = [train_context, train_context_speaker, train_options, train_profile]
    # train_Y = train_target

    ### model.fit(train_X, train_Y, batch_size=hparams.batch_size,
    ###             epochs=hparams.num_epochs,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[tensorboard])#, callbacks=[checkpointer])
   
    # of actual epochs
    val_acc=0
    for i in range(10):
        print('\nMAIN EPOCH:', i+1)
        print('==================================================================================================')
        idx = random.sample(range(100000), 100000)
        for j in range(10):
            print('Sub epochs:', j+1)
            sub_idx = idx[j*10000 : (j+1)*(10000)]
            train_X = [np.take(train_context, sub_idx, axis=0),\
                        # np.take(train_context_speaker, sub_idx, axis=0),\
                        np.take(train_context_mask, sub_idx, axis=0),\
                        np.take(train_options, sub_idx, axis=0)]#,
                        #np.take(train_profile, sub_idx, axis=0)]
            train_Y = np.take(train_target, sub_idx, axis=0)

            A = model.fit(train_X, [train_Y,
                                    np.zeros((10000,1,1), dtype='i4'),
                                    np.zeros((10000,1,1,1,1), dtype='i4'),
                                    np.zeros((10000,1,1,1,1), dtype='i4')],
                                    batch_size=hparams.batch_size,
                                    epochs=1,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[checkpointer])
            # for key in A.history.keys():
            #     print(key)
            
            if A.history['val_probs_acc'][0] > val_acc:
                val_acc = A.history['val_probs_acc'][0]
                if val_acc >= 0.19:
                    model.save_weights(hparams.weights_path+\
                                        'ubuntu_track1_'+\
                                        str(int(hparams.hops))+'hops_'+\
                                        str(int(hparams.learning_rate*10000))+'lr_'+\
                                        str(int(hparams.memn2n_rnn_dim))+'rnn_'+\
                                        str(int(val_acc*10000))+'_'+\
                                        str(i)+'_'+str(j)+'.h5', overwrite=True)

    print('Best acc:',val_acc)
    # import pdb; pdb.set_trace()

    '''
    ############################# EVALUATE #############################
    vocab=[]
    with open(hparams.vocab_path,'r') as f:
        A = f.read()
        vocab = A.split('\n')

    model.load_weights('/ext2/dstc7/weights/memLstm2_bicon4_amp2_nobias/3hops_3_9_1828_5amp_bicon4_amp2_nobias.h5')
    score=model.evaluate(valid_X, valid_Y)
    print(score)


    predict_X = valid_X
    target_X = valid_target

    predict_Y,\
    context_attention,\
    responses_attention,\
    responses_dot = model.predict(predict_X, batch_size=50, verbose=1)
    predict_target = np.argmax(predict_Y, axis=-1)

    print(predict_Y.shape)
    print(context_attention.shape)
    print(responses_attention.shape)

    # context_argmax = [np.argsort(context_attetion[i][::-1] for i in range(len(predict_Y)))
    context_argmax = np.argsort(context_attention, axis=-1)
    # context_argmax = context_argmax[:,:,hparams.max_context_len-hparams.hops:]
    context_argmax = context_argmax[:,:,hparams.max_context_len-5:]
    context_argmax = np.flip(context_argmax, axis=-1)
    print('context_argmax:',context_argmax.shape)
    responses_attention = np.swapaxes(responses_attention, 1,3)
    responses_argmax = np.argmax(responses_attention, axis=-1)

    sorted_predict_Y = [np.argsort(predict_Y[i])[::-1] for i in range(len(predict_Y))]    
    prediction_set = [(target_X[i],sorted_predict_Y[i][:10]) for i in range(len(predict_Y))]
    # with open('valid_predict10_result.pickle','wb') as f:
    #     pickle.dump(prediction_set, f)    

    correct_sample = 0
    wrong_sample = 0
    for idx,value in enumerate(prediction_set):
        if value[0] == value[1][0]:
            correct_sample +=1
        else:
            wrong_sample +=1

    print(" Among {} samples, model predicted {} samples correct, {} samples wrong.".format(len(predict_Y),correct_sample,wrong_sample))
    
    for i in range(valid_target.shape[0]):
        context = valid_context[i]
        context_att = context_attention[i]
        context_arg = context_argmax[i]
        responses_arg = responses_argmax[i]
        if valid_target[i] == predict_target[i]:
            correct_sample += 1
            print(i, 'sample: (attention on Forward / Backward)')
            # print(context_argmax[i][0])
            # print(valid_context[i][ context_argmax[i][0] ])
            # print(vocab[ valid_context[i][context_argmax[i][0]] ])
            print('Attention: ', [ context_att[0][context_arg[0][j]]\
                                    for j in range(len(context_arg[0])) ], ' / ',
                                 [ context_att[1][context_arg[1][j]]\
                                    for j in range(len(context_arg[1])) ] )
            print('Context :', [ vocab[ context[context_arg[0][j]]-1 ]+\
                                '('+str(context_arg[0][j])+')'\
                                for j in range(len(context_arg[0])) ], ' / ',
                                [ vocab[ context[context_arg[1][j]]-1 ]+\
                                '('+str(context_arg[1][j])+')'\
                                for j in range(len(context_arg[1])) ] )
            print('Response:', [ vocab[ context[responses_arg[predict_target[i]][0][j]]-1 ]+\
                                '('+str(responses_arg[predict_target[i]][0][j])+')'\
                                for j in range(hparams.hops)], ' / ',
                                [ vocab[ context[responses_arg[predict_target[i]][1][j]]-1 ]+\
                                '('+str(responses_arg[predict_target[i]][1][j])+')'\
                                for j in range(hparams.hops)], '\n')

    import pdb; pdb.set_trace()
    # np.save('context_attention.npy', context_attention)
    # np.save('responses_attenntion.npy', responses_attention)
    # np.save('responses_dot.npy', responses_dot)
    '''
if __name__ == "__main__":
    main()
