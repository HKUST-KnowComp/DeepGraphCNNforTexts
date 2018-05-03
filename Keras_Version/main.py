# -*- coding: utf-8 -*-
import os
import numpy as np
#from keras.utils.vis_utils import plot_model
from model2 import gcnn
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from keras import backend as K
from keras.utils import to_categorical
import gc
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


def accu(y_true, y_pred):
    a = K.argmax(y_true,1)
    b = K.argmax(y_pred,1)
    c = K.equal(a,b)
    accuracy = (K.cast(c, K.float32))
    return accuracy


batch_size = 128
depth = 3
mkenerls = [64,64,32]
conv_conf = [2,1]
pooling_conf = ["max",2,2]
bn = False
dropout = True
rate = 0.8
activation = "relu"
conf = [50,300,10] #input size
output_dim = 20

lr = 0.0008
epoch = 200
epoch_cont = 300
data_dic = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'cache','Words2Matrix_{}_{}_{}.h5'.format(conf[0], conf[1], conf[2]))

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

#build model
def build_model():
    model = gcnn(depth, mkenerls, conv_conf, pooling_conf, bn, dropout, rate, activation, conf, output_dim)
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics = ["categorical_accuracy"])
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


def getdata(path):
    print(path)
    h5 = h5py.File(path, 'r')
    datax = h5['datax'].value
    datay = h5['datay'].value
    h5.close()
    return datax,datay

#read data
def read_data():
    X_train,Y_train = getdata(os.path.join(data_dic, "data", "train.h5"))
    X_valid,Y_valid = getdata(os.path.join(data_dic, "data", "valid.h5"))
    X_test,Y_test = getdata(os.path.join(data_dic, "data", "test.h5"))
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_valid.shape)
    print(Y_test.shape)
    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test
            
def cache(path,X_train,X_valid,X_test,Y_train,Y_valid,Y_test):
    h5 = h5py.File(path, 'w')
    h5.create_dataset('X_train', data=X_train)
    h5.create_dataset('X_valid', data=X_valid)
    h5.create_dataset('X_test', data=X_test)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_valid', data=Y_valid)
    h5.create_dataset('Y_test', data=Y_test)
    h5.close()
    
def read_cache(path):
    h5 = h5py.File(path, 'r')
    X_train = h5['X_train'].value
    X_valid = h5['X_valid'].value
    X_test = h5['X_test'].value
    Y_train = h5['Y_train'].value
    Y_valid = h5['Y_valid'].value
    Y_test = h5['Y_test'].value
    h5.close()
    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test
    
def main():
    if os.path.exists(filepath):
        print("read data from file")
        X_train,X_valid,X_test,Y_train,Y_valid,Y_test = read_cache(filepath)
    else:
        print("read and store data")
        X_train,X_valid,X_test,Y_train,Y_valid,Y_test = read_data()
        cache(filepath,X_train,X_valid,X_test,Y_train,Y_valid,Y_test)
    model = build_model()

    
    

    fname_param = os.path.join(data_dic,'MODEL', 'best2.h5')
    '''
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=5, mode='max')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='max')
    print('=' * 10)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        nb_epoch=epoch,
                        batch_size=batch_size,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    #保存训练最好模型训练细节，此时测试集为验证集
    model.save_weights(fname_param, overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, 'history.pkl'), 'wb'))
    
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=X_train.shape[0],verbose=0)
    print('训练集最好模型进行预测')
    print('Train score: %s' % str(score))
    score = model.evaluate(X_test,Y_test,batch_size=X_test.shape[0],verbose=0)
    print('Test score: %s' % str(score))
    '''


    #fname_param = os.path.join(data_dic, 'MODEL', 'cont.best2.h5')
    model.load_weights(fname_param)
    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(data_dic,'MODEL', 'cont2.best2.h5')
    X_train2 = np.concatenate((X_train,X_valid),axis = 0)
    y_train2 = np.concatenate((Y_train,Y_valid),axis = 0)
    print(X_train2.shape,y_train2.shape)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='max')
    #保存训练最好模型训练细节，此时训练集+验证集为新的训练集，测试集为测试集
    history = model.fit(X_train2, y_train2,
                        nb_epoch=epoch_cont, 
                        verbose=1, 
                        batch_size=batch_size, 
                        callbacks=[model_checkpoint],#early_stopping,model_checkpoint],
                        validation_data=(X_test, Y_test))
    pickle.dump((history.history), open(os.path.join(path_result, 'cont.history.pkl'), 'wb'))
    model.save_weights(fname_param, overwrite=True)
    print('=' * 10)
    print('The best model to predict')
    
    model.load_weights(fname_param)
    score = model.evaluate(X_train2, y_train2, batch_size=X_train2.shape[0],verbose=0)
    print('Train score: %s' % str(score))
    score = model.evaluate(X_test,Y_test,batch_size=X_test.shape[0],verbose=0)
    print('Test score: %s' % str(score))
    
    print('=' * 10)
    print('Done')
    
    gc.collect()
if __name__ == "__main__":
    main()















