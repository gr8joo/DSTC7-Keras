import os
import time
import numpy as np
import keras
import random

from keras import backend as K
from keras import metrics
from keras.callbacks import Callback
import models.helpers as helpers

from hyperparams import create_hyper_parameters

# from models.dual_encoder import dual_encoder_model
from models.dual_encoder_pretrain import dual_encoder_model
# from models.cnn_1d import cnn_1d_model
from models.cnn_1d_pretrain import cnn_1d_model
from models.memn2n import memn2n_model

# Debugging purpose
# from models.dual_encoder_no_embedding import dual_encoder_no_embedding_model

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

class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()

def top2acc(y_true, y_pred, k=2):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top5acc(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top10acc(y_true, y_pred, k=10):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top50acc(y_true, y_pred, k=50):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def main():
    hparams = create_hyper_parameters()

    print("Loading training data")
    train_context = np.load(hparams.train_context_path)
    # train_context_len = np.load(hparam.train_context_len_path)
    train_target = np.load(hparams.train_target_path)
    train_target = train_target.astype(int)
    train_options = np.load(hparams.train_options_path)
    # train_options_len = np.load(hparam.train_context_path)

    train_X = [train_context, train_options]
    train_Y = train_target


    print("Loading validation data")
    valid_context = np.load(hparams.valid_context_path)
    # valid_context_len = np.load(hparam.valid_context_len_path)
    valid_target = np.load(hparams.valid_target_path)
    valid_target = valid_target.astype(int)
    valid_options = np.load(hparams.valid_options_path)
    # valid_options_len = np.load(hparam.valid_context_path)

    valid_X = [valid_context, valid_options]
    valid_Y = valid_target


    context = Input(shape=(hparams.max_context_len,))
    utterances = Input(shape=(hparams.num_utterance_options, hparams.max_utterance_len))
    inputs = [context, utterances]
    
    # probs = dual_encoder_model(hparams, context, utterances)
    probs = cnn_1d_model(hparams, context, utterances)
    # probs = memn2n_model(hparams, context, utterances)

    model = Model(inputs=inputs, outputs=probs)
    print("Model loaded")

    # out_batch = NBatchLogger(10)
    checkpointer = ModelCheckpoint(filepath='./dual_encoder_checkpoint.h5', verbose=1, save_best_only=True)
    optim = keras.optimizers.Adam(lr=hparams.learning_rate, clipnorm=hparams.clip_norm)
    model.compile(loss={'probs': 'sparse_categorical_crossentropy'},#custom_loss(probs=probs),#{'probs': custom_loss},#
                    optimizer=optim,
                    loss_weights={'probs': 1.0}, metrics=['accuracy', top2acc, top5acc, top10acc, top50acc])

    model.summary()
    model.fit(train_X, train_Y, batch_size=hparams.batch_size,
    	        epochs=hparams.num_epochs,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[checkpointer])
    '''
    for i in range(100):
        indices = random.sample(range(100000), 10000)
        train_X = [np.array(np.take(train_context, indices, axis=0)), np.array(np.take(train_options, indices, axis=0))]
        train_Y = np.take(train_target, indices, axis=0)
        model.fit(train_X, train_Y, batch_size=hparams.batch_size,
            epochs=1,validation_data=(valid_X, valid_Y), verbose=1)#, callbacks=[checkpointer])
    '''

    # model.save_weights('./dual_encoder_weights.h5', overwrite=True)

if __name__ == "__main__":
    main()
