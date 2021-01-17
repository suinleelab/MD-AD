from keras import Model
from keras.layers import Dense, Dropout
from keras.optimizers import adam
import tensorflow as tf
from keras import backend as K
from keras import Input
from keras import regularizers
import numpy as np


def ignorenans_mse(y_true, y_pred):
    bool_finite = tf.is_finite(y_true)
    return K.mean(K.square(tf.boolean_mask(y_pred, bool_finite) - tf.boolean_mask(y_true, bool_finite)), axis=-1)


### some custom loss functions we don't use anymore but are needed 
def ignorenans_categorical_accuracy(num_categories):
    def cat_acc(y_true,y_pred):
        return
    return 

def ordloss(num_categories):
    def ordloss_cur_params(y_true, y_pred):
        return 
    return 

def ignorenans_scaled_mse(y_true, y_pred):
    bool_finite = tf.is_finite(y_true)
    mse =  K.mean(K.square(tf.boolean_mask(y_pred, bool_finite) - tf.boolean_mask(y_true, bool_finite)), axis=-1)
    var = K.var(tf.boolean_mask(y_true, bool_finite), axis=-1)
    return mse/var



######


def MDAD_model(X_train, hyperparams):

    k_reg = hyperparams["k_reg"]
    nonlinearity = hyperparams["nonlinearity"]
    dropout = hyperparams["dropout"]
    grad_clip_norm = hyperparams["grad_clip_norm"]
    learning_rate = hyperparams["learning_rate"]
    hidden_sizes_shared = hyperparams["hidden_sizes_shared"]
    hidden_sizes_separate = hyperparams["hidden_sizes_separate"]
    phenotypes = ["CERAD", "BRAAK", "PLAQUES", "TANGLES", "ABETA_IHC", "TAU_IHC"]
    
    shared_layers = []
    separate_layers = {}
    loss_dict = {}

    for p in phenotypes:
        separate_layers[p] = []
        loss_dict['%s_out'%p] = ignorenans_mse
    
    
    main_input = Input(shape=(X_train.shape[1],), dtype='float', name='main_input')

    # Add hidden SHARED layers 
    for i,s in enumerate(hidden_sizes_shared):
        if i==0:
            shared_layers.append(Dense(s, activation=nonlinearity, kernel_regularizer=regularizers.l2(k_reg), name="Shared_dense_%i"%i)(main_input))
        else:
            shared_layers.append(Dense(s, activation=nonlinearity, kernel_regularizer=regularizers.l2(k_reg), name="Shared_dense_%i"%i)(shared_layers[-1]))
        shared_layers.append(Dropout(dropout, name="Shared_dropout_%i"%i)(shared_layers[-1]))

    # Add hidden SEPARATE layers for each phenotype + Final output layer
    for p in phenotypes:
        for i,s in enumerate(hidden_sizes_separate):
            if i==0:
                separate_layers[p].append(Dense(s, activation=nonlinearity,  kernel_regularizer=regularizers.l2(k_reg), name="%s_dense_%i"%(p,i))(shared_layers[-1]))
            else:
                separate_layers[p].append(Dense(s, activation=nonlinearity, kernel_regularizer=regularizers.l2(k_reg), name="%s_dense_%i"%(p,i))(separate_layers[p][-1]))
            separate_layers[p].append(Dropout(dropout, name="%s_dropout_%i"%(p,i))(separate_layers[p][-1]))
        separate_layers[p].append(Dense(1, name='%s_out'%p)(separate_layers[p][-1]))

    model = Model(inputs=[main_input], outputs=[separate_layers[p][-1] for p in phenotypes])

    opt = adam(clipnorm=grad_clip_norm, lr=learning_rate)

    
    model.compile(optimizer=opt,
                  loss=loss_dict,
                  loss_weights=[1]*len(phenotypes))

    return model



######################


def single_MLP_model(X_train, hyperparams):

    k_reg = hyperparams["k_reg"]
    nonlinearity = hyperparams["nonlinearity"]
    dropout = hyperparams["dropout"]
    grad_clip_norm = hyperparams["grad_clip_norm"]
    learning_rate = hyperparams["learning_rate"]
    hidden_sizes_shared = hyperparams["hidden_sizes_shared"]
    hidden_sizes_separate = hyperparams["hidden_sizes_separate"]

    layers = []

    main_input = Input(shape=(X_train.shape[1],), dtype='float', name='main_input')

    # Add hidden layers -- based on MTL model, build a single task model (including the same number of "shared" and "separate" layers)
    for i,s in enumerate(np.hstack([hidden_sizes_shared, hidden_sizes_separate])):
        if i==0:
            layers.append(Dense(s, activation=nonlinearity, kernel_regularizer=regularizers.l2(k_reg), name="Dense_%i"%i)(main_input))
        else:
            layers.append(Dense(s, activation=nonlinearity, kernel_regularizer=regularizers.l2(k_reg), name="Dense_%i"%i)(layers[-1]))
        layers.append(Dropout(dropout, name="Dropout_%i"%i)(layers[-1]))

    out = Dense(1, name='out')(layers[-1])
    
    model = Model(inputs=[main_input], outputs=[out])

    opt = adam(clipnorm=grad_clip_norm, lr=learning_rate)  

    model.compile(optimizer=opt, loss={'out': ignorenans_mse})

    return model




def single_linear_model(X_train, hyperparams):
    k_reg = hyperparams["k_reg"]
    grad_clip_norm = hyperparams["grad_clip_norm"]
    learning_rate = hyperparams["learning_rate"]

    main_input = Input(shape=(X_train.shape[1],), dtype='float', name='main_input')
    out = Dense(1, kernel_regularizer=regularizers.l2(k_reg), name='out')(main_input)
    model = Model(inputs=[main_input], outputs=[out])

    opt = adam(clipnorm=grad_clip_norm, lr=learning_rate)  
    model.compile(optimizer=opt, loss={'out': ignorenans_mse})

    return model

