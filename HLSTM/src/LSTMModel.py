#-*- coding: UTF-8 -*- 
from datetime import datetime
from EmbLayer import EmbLayer
from LSTMLayer import LSTMLayer
from HiddenLayer import HiddenLayer
from PoolLayer import *
from SentenceSortLayer import *
import theano
import theano.tensor as T
import numpy 
import random
import sys
import time
from Update import AdaUpdates

EVALUTION_THRESHOLD_FOR_MULTI_LABEL = 0.5 # the evalution threshold for multi-label classification

class LSTMModel(object):
    def __init__(self, n_voc, trainset, testset, dataname, classes, prefix):
        if prefix != None:
            prefix += '/'
        self.trainset = trainset
        self.testset = testset
        self.classes = int(classes)

        docs = T.imatrix()
        label = T.imatrix()
        length = T.fvector()
        wordmask = T.fmatrix()
        sentencemask = T.fmatrix()
        maxsentencenum = T.iscalar()
        sentencenum = T.fvector()
        isTrain = T.iscalar()

        rng = numpy.random

        # layers = []
        # layers.append(EmbLayer(rng, docs, n_voc, 50, 'emblayer', dataname, prefix))
        # layers.append(LSTMLayer(rng, layers[-1].output, wordmask, 50, 50, 'wordlstmlayer', prefix))
        # layers.append(SimpleAttentionLayer(rng, layers[-1].output, wordmask,50, 50, 'wordattentionlayer', prefix))
        # layers.append(SentenceSortLayer(layers[-1].output,maxsentencenum,prefix))
        # layers.append(LSTMLayer(rng, layers[-1].output, sentencemask, 50, 50, 'sentencelstmlayer', prefix))
        # layers.append(SimpleAttentionLayer(rng, layers[-1].output, sentencemask,50, 50, 'sentenceattentionlayer', prefix))
        # layers.append(HiddenLayer(rng, layers[-1].output, 50, 50, 'fulllayer', prefix))
        # layers.append(HiddenLayer(rng, layers[-1].output, 50, int(classes), 'softmaxlayer', prefix, activation=T.nnet.sigmoid))
        # self.layers = layers
        layers = []
        layers.append(EmbLayer(rng, docs, n_voc, 50, 'emblayer', dataname, prefix))
        layers.append(LSTMLayer(rng, layers[-1].output, wordmask, 50, 50, 'wordlstmlayer', prefix)) 
        layers.append(MeanPoolLayer(layers[-1].output, length))
        layers.append(SentenceSortLayer(layers[-1].output,maxsentencenum))
        layers.append(LSTMLayer(rng, layers[-1].output, sentencemask, 50, 50, 'sentencelstmlayer', prefix))
        layers.append(MeanPoolLayer(layers[-1].output, sentencenum))
        layers.append(HiddenLayer(rng, layers[-1].output, 50, 50, 'fulllayer', prefix))
        layers.append(HiddenLayer(rng, layers[-1].output, 50, int(classes), 'softmaxlayer', prefix, activation=T.nnet.sigmoid))
        self.layers = layers
        
        predict = layers[-1].output
        cost = T.nnet.binary_crossentropy(layers[-1].output, label).sum(1)
        cost = cost.mean()
        # modifu corrrect.
        # predicted_value = ((layers[-1].output) >= EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)
        # predicted_value = predicted_value.astype(bool)
        # true_value = label.astype(bool)
        # equal = true_value == predicted_value
        # match = np.sum(equal, axis=1) == np.size(equal, axis=1)
        # # value 1 match_ratio
        # exact_match_ratio = np.sum(match) / np.size(match)
        # true_and_predict = np.sum(true_value & predicted_value, axis=1)
        # true_or_predict = np.sum(true_value | predicted_value, axis=1)
        # # value 2 accuracy
        # accuracy = np.mean(true_and_predict / true_or_predict)
        # # value 3 pression
        # precison = np.mean(true_and_predict / (np.sum(predicted_value, axis=1) + 1e-9))
        # # recall 4 recall
        # recall = np.mean(true_and_predict / np.sum(true_value, axis=1))
        # # f1_Measure
        # F1_Measure = np.mean((true_and_predict * 2) / (np.sum(true_value, axis=1) + np.sum(predicted_value, axis=1)))
        # # HammingLoss
        # HammingLoss = np.mean(true_value ^ total_predicted_value)
        # TP
        # TP = np.sum(true_value & predicted_value,axis=0,dtype=np.int32)
        # FP = np.sum((~true_value) & predicted_value,axis=0,dtype=np.int32)
        # FN = np.sum(true_value & (~predicted_value),axis=0,dtype=np.int32)
        # _P = np.sum(TP) / (np.sum(TP) + np.sum(FP)  + 1e-9 )
        # _R = np.sum(TP) / (np.sum(TP) + np.sum(FN)  + 1e-9 )
        # Micro_F1 = (2 * _P *_R) / (_P + _R)
        # _P_t = TP / (TP + FP + 1e-9)
        # _R_t = TP / (TP + FN + 1e-9)
        # Macro_F1 = np.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-9))
        #cost = -T.mean(T.log(layers[-1].output)[T.arange(label.shape[0]), label], acc_dtype='float32')
        #modify this
        #correct = T.sum(T.eq(T.argmax(layers[-1].output, axis=1), label), acc_dtype='int32')
        #err = T.argmax(layers[-1].output, axis=1) - label
        #mse = T.sum(err * err)
        
        params = []
        for layer in layers:
            params += layer.params
        L2_rate = numpy.float32(1e-5)
        for param in params[1:]:
            cost += T.sum(L2_rate * (param * param), acc_dtype='float32')
        gparams = [T.grad(cost, param) for param in params]

        updates = AdaUpdates(params, gparams, 0.95, 1e-6)

        self.train_model = theano.function(
            inputs=[docs, label,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=cost,
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[docs,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=predict,
        )

    def train(self, iters):
        lst = numpy.random.randint(self.trainset.epoch, size = iters)
        n = 0
        for i in lst:
            n += 1
            out = self.train_model(self.trainset.docs[i], self.trainset.label[i], self.trainset.length[i],self.trainset.sentencenum[i],self.trainset.wordmask[i],self.trainset.sentencemask[i],self.trainset.maxsentencenum[i])
            print n, 'cost:', out, 'time', datetime.now()
        
    def test(self):
        file_eval = open('evallog.txt','a')
        old = sys.stdout
        sys.stdout = file_eval
        print 'time start:', datetime.now()
        sys.stdout = old
        total_predicted_value = numpy.zeros([1, self.classes], dtype=numpy.float32)  ##
        total_true_value = numpy.zeros([1, self.classes], dtype=numpy.int32)
        for i in xrange(self.testset.epoch):
            predicted_value = self.test_model(self.testset.docs[i],self.testset.length[i], self.testset.sentencenum[i], self.testset.wordmask[i],self.testset.sentencemask[i],self.testset.maxsentencenum[i])
            total_predicted_value = numpy.concatenate((total_predicted_value, predicted_value), axis=0)
            total_true_value = numpy.concatenate((total_true_value, self.testset.label[i]), axis=0)
        total_predicted_value = total_predicted_value[1:]
        total_true_value = total_true_value[1:]
        assert len(total_true_value) == len(total_predicted_value), 'shape error' 
        total_predicted_value = ((total_predicted_value) >= EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)
        total_predicted_value = total_predicted_value.astype(bool)
        total_true_value = total_true_value.astype(bool)
        TP = numpy.sum(total_true_value & total_predicted_value,axis=0,dtype=numpy.int32)
        FP = numpy.sum((~total_true_value) & total_predicted_value,axis=0,dtype=numpy.int32)
        FN = numpy.sum(total_true_value & (~total_predicted_value),axis=0,dtype=numpy.int32)
        _P = numpy.sum(TP) / (numpy.sum(TP) + numpy.sum(FP)  + 1e-9 )
        _R = numpy.sum(TP) / (numpy.sum(TP) + numpy.sum(FN)  + 1e-9 )
        Micro_F1 = (2 * _P *_R) / (_P + _R + 1e-9)
        _P_t = TP / (TP + FP + 1e-9)
        _R_t = TP / (TP + FN + 1e-9)
        print 'TP',TP,'FP',FP,'FN',FN
        Macro_F1 = numpy.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-9))
        print('Micro-F1 = %.4f' % Micro_F1)
        print('Macro-F1 = %.4f' % Macro_F1)
        old = sys.stdout
        sys.stdout = file_eval
        print 'time end:', datetime.now()
        print 'TP',TP,'FP',FP,'FN',FN
        print('Micro-F1 = %.4f' % Micro_F1)
        print('Macro-F1 = %.4f' % Macro_F1)
        sys.stdout = old
        file_eval.close()
        return Micro_F1, Macro_F1


    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)
