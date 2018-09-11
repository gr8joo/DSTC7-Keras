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

from models.memLstm2_bicon3_KB import memLstm_KB_model

from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

def custom_loss(probs):
    def loss(y_true, y_pred):
        mse_loss = -K.sum(K.square(probs))
        # K.update_add(mse_loss, 2*K.square(probs[y_true]))
        mse_loss = mse_loss + 2*K.square(probs[y_true])
        return mse_loss
    return loss


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
    context_speaker = Input(shape=(hparams.max_context_len, 2))
    utterances = Input(shape=(hparams.num_utterance_options, hparams.max_utterance_len))
    knowledges = Input(shape=(hparams.num_knowledge_options, hparams.max_knowledge_len))
    # profile = Input(shape=(hparams.max_profile_len,))


    inputs = [context, context_speaker, utterances, knowledges]
    # inputs = [context, context_speaker, utterances, profile]
    

    # probs = dual_encoder_model(hparams, context, context_speaker, utterances)
    # probs = dual_encoder_model(hparams, context, context_speaker, utterances, profile)
    # probs = cnn_1d_model(hparams, context, context_speaker, utterances)
    probs = memLstm_KB_model(hparams, context, context_speaker, utterances, knowledges)
    # probs = memLstm_model(hparams, context, context_speaker, utterances, profile)


    model = Model(inputs=inputs, outputs=probs)
    print("Model loaded")


    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    tensorboard =\
    TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # checkpointer =\
    # ModelCheckpoint(filepath='./dual_encoder_checkpoint.h5', verbose=1, save_best_only=True)
    # optim = keras.optimizers.SGD(lr=hparams.learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    optim = keras.optimizers.Adam(lr=hparams.learning_rate, clipnorm=hparams.clip_norm)#, decay=0.001)
    model.compile(loss={'probs': 'sparse_categorical_crossentropy'},#custom_loss(probs=probs),#{'probs': custom_loss},
                    optimizer=optim,
                    # loss_weights={'probs': 1.0},
                    metrics=['accuracy', top2acc, top5acc, top10acc, top50acc])
    model.summary()


    ############################# Load Datas #############################    
    print("Loading training data")
    train_context = np.load(hparams.train_context_path)
    train_context_speaker =np.load(hparams.train_context_speaker_path)
    # train_context_len = np.load(hparam.train_context_len_path)
    train_target = np.load(hparams.train_target_path)
    train_target = train_target.astype('i4')
    train_options = np.load(hparams.train_options_path)
    # train_options_len = np.load(hparam.train_context_path)
    # train_profile = np.load(hparams.train_profile_path)

    print("Loading validation data")
    valid_context = np.load(hparams.valid_context_path)
    valid_context_speaker =np.load(hparams.valid_context_speaker_path)
    # valid_context_len = np.load(hparam.valid_context_len_path)
    valid_target = np.load(hparams.valid_target_path)
    valid_target = valid_target.astype('i4')
    valid_options = np.load(hparams.valid_options_path)
    # valid_options_len = np.load(hparam.valid_context_path)
    # valid_profile = np.load(hparams.valid_profile_path)



    # train_X = [train_context, train_context_speaker, train_options]
    # train_X = [train_context, train_context_speaker, train_options, train_profile]
    # train_Y = train_target
    
    valid_X = [valid_context, valid_context_speaker, valid_options]
    # valid_X = [valid_context, valid_context_speaker, valid_options, valid_profile]
    valid_Y = valid_target


    ############################# TRAIN #############################
    ### model.fit(train_X, train_Y, batch_size=hparams.batch_size,
    ### 	        epochs=hparams.num_epochs,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[tensorboard])#, callbacks=[checkpointer])
   
    # of actual epochs
    val_acc=0
    for i in range(50):
        print('\nMAIN EPOCH:', i+1)
        print('==================================================================================================')
        idx = random.sample(range(100000), 100000)
        for j in range(10):
            print('Sub epochs:', j+1)
            sub_idx = idx[j*10000 : (j+1)*(10000)]
            train_X = [np.take(train_context, sub_idx, axis=0),\
                        np.take(train_context_speaker, sub_idx, axis=0),\
                        np.take(train_options, sub_idx, axis=0)]#,
                        #np.take(train_profile, sub_idx, axis=0)]
            train_Y = np.take(train_target, sub_idx, axis=0)
            A = model.fit(train_X, train_Y, batch_size=hparams.batch_size,
                            epochs=1,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[checkpointer])
            # for key in A.history.keys():
            #     print(key)
            if A.history['val_acc'][0] > val_acc:
                val_acc = A.history['val_acc'][0]
            if A.history['val_acc'][0] >= 0.1800:
                model.save_weights(hparams.weights_path+'_'+str(i)+'_'+str(j)+'_'+str(int(A.history['val_acc'][0]*10000))+'.h5', overwrite=True)


    print('Best acc:',val_acc)


    '''
    ############################# EVALUATE #############################
    # model.load_weights('weights/memLstm2_bicon2_profile/2hops_3_5_1180.h5')
    model.load_weights('weights/13hops_2_9_1807.h5')
    score=model.evaluate(valid_X, valid_Y)
    print(score)


    predict_X = valid_X
    target_X = valid_target

    predict_Y = model.predict(predict_X, batch_size=10, verbose=1)    
    sorted_predict_Y = [np.argsort(predict_Y[i])[::-1] for i in range(len(predict_Y))]    
    prediction_set = [(target_X[i],sorted_predict_Y[i][:10]) for i in range(len(predict_Y))]
    with open('valid_predict10_result.pickle','wb') as f:
        pickle.dump(prediction_set, f)    

    correct_sample = 0
    wrong_sample = 0
    for idx,value in enumerate(prediction_set):
        if value[0] == value[1][0]:
            correct_sample +=1
        else:
            wrong_sample +=1

    print(" Among {} samples, model predicted {} samples correct, {} samples wrong.".format(len(predict_Y),correct_sample,wrong_sample))
    '''
if __name__ == "__main__":
    main()
