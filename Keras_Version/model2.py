# -*- coding: utf-8 -*-
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Flatten,
    Dense,
    Reshape)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.pooling import MaxPool2D, AveragePooling2D
import time


def mpool(type, input, size, stride):
    if type == "max":
        return MaxPool2D(pool_size=(size, size), strides=stride, padding='same')(input)
    elif type == "avg":
        return AveragePooling2D(pool_size=(size, size), strides=stride, padding='same')(input)
    else:
        raise ValueError("pooling type invalid")


def active(type, input):
    if type == "relu":
        return Activation("relu")(input)
    elif type == "sigmoid":
        return Activation("sigmoid")(input)
    elif type == "tanh":
        return Activation("tanh")(input)
    elif type == "softmax":
        return Activation("softmax")(input)
    else:
        raise ValueError("activation type invalid")


def gcnn(depth=4, mkenerls=[64, 64, 64, 32], conv_conf=[2, 1], pooling_conf=["max", 2, 2], bn=False, dropout=True,
         rate=0.8, activation="relu", conf=[50, 300, 10], output_dim=20):
    assert depth == len(mkenerls)
    mchannel, mheight, mwidth = conf
    conv_size, conv_stride = conv_conf
    pooling_type, pooling_size, pooling_stride = pooling_conf
    input = Input(shape=(mchannel, mheight, mwidth))

    conv1 = Convolution2D(filters=mkenerls[0], kernel_size=(1, mwidth), strides=(1, 1), padding="valid")(input)
    # bn1 = BatchNormalization(axis=1)(conv1)
    activation1 = Activation("relu")(conv1)
    pool1 = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(activation1)
    _k1, _n1 = map(int, pool1.shape[1:3])
    reshape_pool1 = Reshape((1, _k1, _n1))(pool1)

    conv2 = Convolution2D(filters=mkenerls[1], kernel_size=(1, _n1), strides=(1, 1), padding="valid")(reshape_pool1)
    # bn2 = BatchNormalization(axis=1)(conv2)
    activation2 = Activation("relu")(conv2)
    pool2 = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(activation2)
    _k2, _n2 = map(int, pool2.shape[1:3])
    reshape_pool2 = Reshape((1, _k2, _n2))(pool2)

    conv3 = Convolution2D(filters=mkenerls[1], kernel_size=(1, _n2), strides=(1, 1), padding="valid")(reshape_pool2)
    # bn2 = BatchNormalization(axis=1)(conv2)
    activation3 = Activation("relu")(conv3)
    pool3 = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(activation3)
    _k3, _n3 = map(int, pool2.shape[1:3])
    reshape_pool3 = Reshape((1, _k2, _n2))(pool3)

    conv4 = Convolution2D(filters=mkenerls[2], kernel_size=(1, _n3), strides=(1, 1), padding="valid")(reshape_pool2)
    # bn3 = BatchNormalization(axis=1)(conv3)
    activation4 = Activation("relu")(conv4)
    pool4 = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(activation4)

    # step_results = [input]
    # for i in range(depth - 1):
    #     mconv = Convolution2D(
    #         nb_filter=mkenerls[i], nb_row=conv_size, nb_col=conv_size, strides=(conv_stride, conv_stride),
    #         border_mode="same")(step_results[-1])
    #     if bn:
    #         mbn = BatchNormalization(axis=1)(mconv)
    #     else:
    #         mbn = mconv
    #     mactivation = active(activation, mbn)
    #     mpooling = mpool(pooling_type, mactivation, pooling_size, pooling_stride)
    #     if dropout:
    #         mdropout = Dropout(rate=rate, seed=time.time())(mpooling)
    #     else:
    #         mdropout = mpooling
    #     step_results.append(mdropout)

    # last_conv = Convolution2D(
    #     nb_filter=mkenerls[-1], nb_row=conv_size, nb_col=conv_size, border_mode="same")(step_results[-1])
    # last_pooling = mpool(pooling_type, last_conv, pooling_size, pooling_stride)
    mFlatten = Flatten()(pool4)
    ms_output = Dense(output_dim=128)(mFlatten)
    msinput = active("sigmoid", ms_output)
    moutput = Dense(output_dim=output_dim)(msinput)
    output = active("softmax", moutput)
    model = Model(input=input, output=output)
    return model


if __name__ == '__main__':
    model = gcnn()
    model.summary()

