import os
import time
import numpy as np
import keras
import random
import pickle
import sys
import itertools

from keras import backend as K
from keras import metrics
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
import models.helpers as helpers

from hyperparams import create_hyper_parameters

from models.memLstm2_advising import memLstm_custom_model

from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import ray
import ray.tune as tune
from ray.tune import grid_search, run_experiments
from ray.tune import Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

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


class memLstm(Trainable):
    def _read_data(self):

        hparams = self.hparams
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
        valid_context_profile_flag = np.load(hparams.valid_context_profile_flag_path)
        valid_options_profile_flag = np.load(hparams.valid_options_profile_flag_path)


        valid_context_mask = hparams.neg_inf * valid_context_mask

    
        self.valid_X = [valid_context, valid_context_mask,
                        valid_options,
                        valid_context_profile_flag, valid_options_profile_flag]        
        # valid_X = [valid_context, valid_context_speaker, valid_options, valid_profile]
        self.valid_Y = [valid_target, np.zeros((500,1,1),dtype='i4'),
                                    np.zeros((500,1,1,1,1), dtype='i4'),
                                    np.zeros((500,1,1,1,1), dtype='i4')]


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
        train_context_profile_flag = np.load(hparams.train_context_profile_flag_path)
        train_options_profile_flag = np.load(hparams.train_options_profile_flag_path)



        train_context_mask = hparams.neg_inf * train_context_mask

        self.train_context = train_context
        self.train_context_mask = train_context_mask
        self.train_options = train_options
        self.train_target = train_target
        self.train_context_profile_flag = train_context_profile_flag
        self.train_options_profile_flag = train_options_profile_flag
    
    def _build_model(self):

        hparams = self.hparams

        ############### Load a model and shaping model input&output ###############
        context = Input(shape=(hparams.max_context_len,))
        # context_speaker = Input(shape=(hparams.max_context_len, 2))
        context_mask = Input(shape=(hparams.max_context_len,))
        utterances = Input(shape=(hparams.num_utterance_options, hparams.max_utterance_len))
        
        context_profile_flag = Input(shape=(hparams.max_context_len, hparams.max_profile_len))
        utterances_profile_flag = Input(shape=(hparams.num_utterance_options, hparams.max_profile_len))


        inputs = [context, context_mask,
                    utterances,
                    context_profile_flag, utterances_profile_flag]
        

        probs,\
        context_attention,\
        responses_attention,\
        responses_dot = memLstm_custom_model(hparams, context, context_mask,
                                                        utterances,
                                                        context_profile_flag, utterances_profile_flag)
        # probs = memLstm_model(hparams, context, context_speaker, utterances, profile)


        model = Model(inputs=inputs, outputs=[probs,
                                                context_attention,
                                                responses_attention,
                                                responses_dot])
        print("Model loaded")
        return model

    def _setup(self):
        self.val_acc = 0
        self.main_epoch = 0
        self.hparams = create_hyper_parameters()

        self.hparams.learning_rate = self.config['learning_rate']
        self.hparams.memn2n_rnn_dim = int(self.config['memn2n_rnn_dim'])
        self.hparams.hops = int(self.config['hops'])
        self.hparams.amp_val = int(self.config['amp'])

        self._read_data()
        model = self._build_model()


         # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        tensorboard =\
        TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        # checkpointer =\
        # ModelCheckpoint(filepath='./dual_encoder_checkpoint.h5', verbose=1, save_best_only=True)
        # optim = keras.optimizers.SGD(lr=hparams.learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        optim = keras.optimizers.Adam(lr=self.hparams.learning_rate, clipnorm=self.hparams.clip_norm, decay=self.hparams.learning_rate*0.0001)
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
        self.model = model

    def _train(self):

        hparams = self.hparams

        train_context = self.train_context
        train_context_mask = self.train_context_mask
        train_options = self.train_options
        train_target = self.train_target
        train_context_profile_flag = self.train_context_profile_flag
        train_options_profile_flag = self.train_options_profile_flag

        valid_X = self.valid_X
        valid_Y = self.valid_Y

        
        idx = random.sample(range(100000),100000)
        for j in range(10):        
            print("Sub epochs", j+1)
            sub_idx = idx[j*10000 : (j+1)*(10000)]
            train_X = [np.take(train_context, sub_idx, axis=0),
                        np.take(train_context_mask, sub_idx, axis=0),\
                        np.take(train_options, sub_idx, axis=0),
                        np.take(train_context_profile_flag, sub_idx, axis=0),
                        np.take(train_options_profile_flag, sub_idx, axis=0)]
            train_Y = np.take(train_target, sub_idx, axis=0)

            A = self.model.fit(train_X, [train_Y,
                                    np.zeros((10000,1,1), dtype='i4'),
                                    np.zeros((10000,1,1,1,1), dtype='i4'),
                                    np.zeros((10000,1,1,1,1), dtype='i4')],
                                    batch_size=hparams.batch_size,
                                    epochs=1,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[checkpointer])
            
            if A.history['val_probs_acc'][0] > self.val_acc:
                self.val_acc = A.history['val_probs_acc'][0]
                if self.val_acc >= 0.09:
                    self.model.save_weights(hparams.weights_path+\
                                        'advising_task1_'+\
                                        str(int(hparams.hops))+'hops_'+\
                                        str(int(hparams.learning_rate*10000))+'lr_'+\
                                        str(int(hparams.memn2n_rnn_dim))+'rnn_'+\
                                        str(int(self.val_acc*10000))+'_'+\
                                        str(int(hparams.amp_val))+'_'+\
                                        str(self.main_epoch)+'_'+str(j)+'.h5', overwrite=True)
        
        self.main_epoch += 1
        # accuracy = self.model.evaluate(valid_X, valid_Y)
        # mean_accuracy = accuracy[5]
        return {"mean_accuracy": self.val_acc}

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save_weights(file_path)
        return file_path

    def _restore(self, path):
        self.model.load_weights(path)

    def _stop(self):
        pass
        

if __name__ == "__main__":
    from hyperopt import hp
    
    tune.register_trainable("my_class", memLstm)
    ray.init(redis_address="192.168.1.153:9023")


    space={
        'learning_rate': hp.uniform('learning_rate', 0.0005, 0.0001),
        'memn2n_rnn_dim': hp.uniform('memn2n_rnn_dim', 128, 257),
        'hops': hp.choice('hops', [3,4,5,6]),
        'amp': hp.choice('amp', [1,2,3,4,5])
    }

    config = {
        'my_exp': {
            'run': memLstm,
            'trial_resources': {'gpu': 1},
            'stop': {
                "training_iteration": 5
            },       
            'num_samples':8
        }
    }

    algo = HyperOptSearch(space, max_concurrent=4, reward_attr="mean_accuracy")
    scheduler = AsyncHyperBandScheduler(reward_attr="mean_accuracy")
    
    run_experiments(config, search_alg = algo, scheduler=scheduler)

