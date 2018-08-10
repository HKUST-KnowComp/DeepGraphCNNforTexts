"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from  torch.nn import DataParallel
os.environ["CUDA_VISIBLE_DEVICES"]="1"
devices = [1,2]

BATCH_SIZE = 32
NUM_CLASSES = 103
NUM_EPOCHS = 200
GPU = True
load =False


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        #   96,10,50
        self.conv1 = nn.Conv2d(in_channels=50, out_channels=64, kernel_size=3, stride=1)
        #   94  8 64
        self.pooling1 = nn.MaxPool2d((2, 1))
        #   47  8  64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,1), stride=1)
        #   45  8  128
        self.pooling2 = nn.MaxPool2d((2, 2))
        #   22  4   128

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,1), stride=1)
        #   20   4    256

        self.pooling3 = nn.MaxPool2d((2, 2))
        #   10  2 256
        self.fc1 = nn.Linear(256*20,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,103)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x =self.pooling1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x =self.pooling2(x)
        x = F.relu(self.conv3(x), inplace=True)
        x =self.pooling3(x)

        x = x.view(x.size(0), -1)


        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        classes = self.fc3(x)
        classes = F.sigmoid(classes)
        return classes


class Mymeter():
    def __init__(self,class_num):
        self.tp = [0]*class_num
        self.fp = [0]*class_num
        self.fn = [0]*class_num
        self.pre = 0.
        self.rec = 0.
        self.class_num = class_num
        
    def process(self,tar,pre):
        for t in tar :
            if t in pre:
                self.tp[t] = self.tp[t]+1
            else:
                self.fn[t] = self.fn[t]+1
        for t in pre :
            if t not in tar:
                self.fp[t] = self.fp[t]+1
    
    def reset(self):
        self.tp = [0]*self.class_num
        self.fp = [0]*self.class_num
        self.fn = [0]*self.class_num
        self.pre = 0.
        self.rec = 0.
    
    
    def micro(self):
        if(sum(self.tp)+sum(self.fp))==0:
            self.pre =0 
        else:
            self.pre = sum(self.tp)/(sum(self.tp)+sum(self.fp))
            
            
        if (sum(self.tp)+sum(self.fn))==0:
            self.rec =0
        else:
            self.rec = sum(self.tp)/(sum(self.tp)+sum(self.fn))
        
        if self.rec==0 and self.pre==0:
            f1 =0
        else:
            f1 = 2*self.pre*self.rec/(self.pre+self.rec)
        return self.pre,self.rec,f1
    
    def macro(self):
        pre = [0.]*self.class_num
        recall = [0.]*self.class_num
        for i in range(self.class_num):
            if (self.tp[i]+self.fp[i]) == 0:
                pre[i]==0.
            else:
                pre[i] = self.tp[i]/(self.tp[i]+self.fp[i])
            
            if (self.tp[i]+self.fn[i]) == 0:
                recall[i]==0.
            else:
                recall[i] = self.tp[i]/(self.tp[i]+self.fn[i])
        
        ma_pre = sum(pre)/self.class_num
        ma_recall =sum(recall)/self.class_num
        if ma_pre+ma_recall==0:
            ma_f1 = 0.
        else:
        
            ma_f1 =  2*ma_pre*ma_recall/(ma_pre+ma_recall)
        return ma_pre,ma_recall,ma_f1

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt
    import h5py
    import os
    import json

    with open('/home/LAB/penghao/mars/metadata/heiring.json','r') as f:
        heir = json.load(f)
    
    model = CapsuleNet()
   # model.load_state_dict(torch.load('epochs/epoch_40.pt'))
    if load :
        saved_state = torch.load('epochs/epoch_49.pt')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in saved_state.items():
            namekey = k[7:]
            new_state_dict[namekey] = v
        model.load_state_dict(new_state_dict)

    if GPU:
        print( torch.cuda.device_count()) 
        model.cuda()
        model = nn.DataParallel(model) 
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    
    mymeter = Mymeter(NUM_CLASSES)
    loss_func = F.binary_cross_entropy

    

    def get_iterator(mode):
        if mode:
            train_path = '/home/LAB/penghao/mars/metadata/train'
            dir = os.listdir(train_path)
            data = None
            labels =None
            flag = 0
            for list in dir:
                f = h5py.File(os.path.join(train_path,list))
                datax = f['datax']
                datax = np.array(datax)
                datay = f['datay']
                datay = np.array(datay)
                datay = datay.astype('float32')
                
                if not flag:
                    data = datax
                    labels = datay
                    flag = 1                    
                else:
                    data = np.concatenate((data,datax), axis=0)
                    labels = np.concatenate((labels,datay),axis=0)
            print ('train set loaded')
            data = data/18.
            tensor_dataset = tnt.dataset.TensorDataset([data, labels])
            return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=16, shuffle=mode)
            
        else:
            train_path = '/home/LAB/penghao/mars/metadata/test'
            dir = os.listdir(train_path)
            data = None
            labels =None
            flag = 0
            for list in dir[:1]:
                f = h5py.File(os.path.join(train_path,list))
                datax = f['datax']
                datax = np.array(datax)
                datay = f['datay']
                datay = np.array(datay)
                datay = datay.astype('float32')
                
                if not flag:
                    data = datax
                    labels = datay
                    flag = 1                    
                else:
                    data = np.concatenate((data,datax), axis=0)
                    labels = np.concatenate((labels,datay),axis=0)
            print ('test set loaded')
            data = data/18.
            tensor_dataset = tnt.dataset.TensorDataset([data, labels])
            return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=8, shuffle=mode)


    def processor(sample):
        data, labels, training = sample

        if GPU:
            data = Variable(data).cuda()
            labels = Variable(labels).cuda()
        labels = labels.float()
        #temp = [np.where(r == 1.)[0][0] for r in labels]
        #temp = torch.LongTensor(temp)
        output = model(data)
        temp = 0
        for i in heir:
            if GPU:
                weight1 = model.module.fc3.weight[i[0]]
                weight2 = model.module.fc3.weight[i[1]]
            else:
                weight1 = model.fc3.weight[i[0]]
                weight2 = model.fc3.weight[i[1]]
            temp =temp+ torch.sum((weight1-weight2)*(weight1-weight2))/2

        loss = loss_func(output, labels) +temp*0.00005

        return loss, output



            

    def reset_meters():
        meter_loss.reset()
        mymeter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        a = state['sample'][1].numpy()
        #计算多标签的参数
        #a为multilabels
        #output为网络结果
        if GPU:
            output = state['output'].data.cpu().numpy()
        else:
            output = state['output'].data.numpy()
        for index in range(a.shape[0]):            #对于Batch中的每个sample
            label = []                             #这个sample中label          
            indices = []            
            for i in range(NUM_CLASSES):
                if a[index][i]==1.0:
                    label.append(i)
                if output[index][i] > 0.5:
                    indices.append(i)
            label = np.array(label)
            indices = np.array(indices)

            mymeter.process(label,indices)
        meter_loss.add(state['loss'].item())            
             

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        mi_pre,mi_rec,mi_f1 = mymeter.micro()
        ma_pre,ma_rec,ma_f1 = mymeter.macro()
        train_loss  = meter_loss.value()[0]
        print ('[Epoch %d] train Loss: %.4f, mi_precision:%.4f  mi_recall:%0.4f  mi_f1:%0.4f    ma_precision:%.4f  ma_recall:%0.4f  ma_f1:%0.4f'%(state['epoch'],train_loss,mi_pre,mi_rec,mi_f1,ma_pre,ma_rec,ma_f1))  
        reset_meters()

        
        if state['epoch']%1 == 0: 
            
            engine.test(processor, get_iterator(False))
            test_mi_pre,test_mi_rec,test_mi_f1 = mymeter.micro()
            test_ma_pre,test_ma_rec,test_ma_f1 = mymeter.macro()
            test_loss  = meter_loss.value()[0]
            print ('[Epoch %d] test Loss: %.4f, mi_precision:%.4f  mi_recall:%0.4f  mi_f1:%0.4f    ma_precision:%.4f  ma_recall:%0.4f  ma_f1:%0.4f'%(state['epoch'],test_loss,test_mi_pre,test_mi_rec,test_mi_f1,test_ma_pre,test_ma_rec,test_ma_f1)) 
            with open('result.txt','a') as f:
                f.write('%d %.4f %.4f %.4f %.4f %.4f %.4f\n' %(state['epoch'],train_loss,mi_f1,ma_f1,test_loss,test_mi_f1,test_ma_f1))     
        else:
            with open('result.txt','a') as f:
                f.write('%d %.4f %.4f %.4f\n' %(state['epoch'],train_loss,mi_f1,ma_f1))    
        
        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])
        
        

    def on_start(state):
        state['epoch'] = 49
    #
    #engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
    t_acc = []
    result =[]
    with open('result_multi.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            line = list(map(eval,line))
            result.append(line)
            t_acc.append(line[4])
    index = t_acc.index(max(t_acc))
    print ('--------')
    print ('The best model to predict')
    print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            result[index][0], result[index][1], result[index][2]))
    print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            result[index][0], result[index][3], result[index][4]))
    print ('--------')
