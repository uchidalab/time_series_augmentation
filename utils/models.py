from keras.models import Model, Input
from keras.layers import Dense, Flatten, Dropout
from keras.layers import MaxPooling1D, Conv1D
from keras.layers import LSTM, Bidirectional
from keras.layers import BatchNormalization, GlobalAveragePooling1D, Permute, concatenate, Activation, add
import numpy as np
import math

def get_model(model_name, input_shape, nb_class):
    if model_name == "vgg":
        model = cnn_vgg(input_shape, nb_class)
    elif model_name == "lstm1":
        model = lstm1(input_shape, nb_class)
    elif model_name == "lstm2":
        model = lstm2(input_shape, nb_class)
    elif model_name == "blstm1":
        model = blstm1(input_shape, nb_class)
    elif model_name == "blstm2":
        model = blstm2(input_shape, nb_class)
    elif model_name == "lstmfcn":
        model = lstm_fcn(input_shape, nb_class)
    elif model_name == "resnet":
        model = cnn_resnet(input_shape, nb_class)
    else:
        model = cnn_lenet(input_shape, nb_class)
    return model

def cnn_lenet(input_shape, nb_class):
    # Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
    
    ip = Input(shape=input_shape)
    
    conv = ip
    
    nb_cnn = int(round(math.log(input_shape[0], 2))-3)
    print("pooling layers: %d"%nb_cnn)
    
    for i in range(nb_cnn):
        conv = Conv1D(6+10*i, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        
    flat = Flatten()(conv)
    
    fc = Dense(120, activation='relu')(flat)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(84, activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    
    out = Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model


def cnn_vgg(input_shape, nb_class):
    # K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
    
    ip = Input(shape=input_shape)
    
    conv = ip
    
    nb_cnn = int(round(math.log(input_shape[0], 2))-3)
    print("pooling layers: %d"%nb_cnn)
    
    for i in range(nb_cnn):
        num_filters = min(64*2**i, 512)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        if i > 1:
            conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        
    flat = Flatten()(conv)
    
    fc = Dense(4096, activation='relu')(flat)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(4096, activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    
    out = Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model

def lstm1(input_shape, nb_class):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
    
    # Hyperparameter choices: 
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    
    ip = Input(shape=input_shape)

    l2 = LSTM(100)(ip)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def lstm2(input_shape, nb_class):
    ip = Input(shape=input_shape)

    l1 = LSTM(100, return_sequences=True)(ip)
    l2 = LSTM(100)(l1)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def blstm1(input_shape, nb_class):
    # Original proposal:
    # M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.
    
    # Hyperparameter choices: 
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    ip = Input(shape=input_shape)

    l2 = Bidirectional(LSTM(100))(ip)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def blstm2(input_shape, nb_class):
    ip = Input(shape=input_shape)

    l1 = Bidirectional(LSTM(100, return_sequences=True))(ip)
    l2 = Bidirectional(LSTM(100))(l1)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def lstm_fcn(input_shape, nb_class):
    # F. Karim, S. Majumdar, H. Darabi, and S. Chen, “LSTM Fully Convolutional Networks for Time Series Classification,” IEEE Access, vol. 6, pp. 1662–1669, 2018.

    ip = Input(shape=input_shape)
    
    # lstm part is a 1 time step multivariate as described in Karim et al. Seems strange, but works I guess.
    lstm = Permute((2, 1))(ip)

    lstm = LSTM(128)(lstm)
    lstm = Dropout(0.8)(lstm)

    conv = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    flat = GlobalAveragePooling1D()(conv)

    flat = concatenate([lstm, flat])

    out = Dense(nb_class, activation='softmax')(flat)

    model = Model([ip], [out])

    model.summary()

    return model


def cnn_resnet(input_shape, nb_class):
    # I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P-A Muller, "Data augmentation using synthetic data for time series classification with deep residual networks," International Workshop on Advanced Analytics and Learning on Temporal Data ECML/PKDD, 2018

    ip = Input(shape=input_shape)
    residual = ip
    conv = ip
    
    for i, nb_nodes in enumerate([64, 128, 128]):
        conv = Conv1D(nb_nodes, 8, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv1D(nb_nodes, 5, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv1D(nb_nodes, 3, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        if i < 2:
            # expands dimensions according to Fawaz et al.
            residual = Conv1D(nb_nodes, 1, padding='same', kernel_initializer="glorot_uniform")(residual)
        residual = BatchNormalization()(residual)
        conv = add([residual, conv])
        conv = Activation('relu')(conv)
        
        residual = conv
    
    flat = GlobalAveragePooling1D()(conv)

    out = Dense(nb_class, activation='softmax')(flat)

    model = Model([ip], [out])

    model.summary()

    return model