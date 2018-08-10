# -*- coding: utf-8 -*-
import os
import zipfile
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import re
import json
import numpy as np
import gensim
import h5py
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import nltk

PATH  = "/home/penghao/mars/rcv2"
original_path = r'/home/penghao/mars/rcv2/reuters/training'
targetpath = r'/data/LJ/LJ/own/RCV1/target_files'
# targetpath = os.path.join(PATH,"target_files")
all = 0
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*','”','“','’',"‘","'",'"']
wordEngStop = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def unzip(file,name):
    global all
    zip_file = zipfile.ZipFile(file)
    path = os.path.join(targetpath,name)
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    for name in zip_file.namelist():
        zip_file.extract(name,path)
        all += 1
    print(all)

def zipp():
    flist = os.listdir(original_path)
    flist.sort()
    for f in flist:
        fname = f.split('.')[0]
        print(fname)
        fpath = os.path.join(original_path,f)
        print(fpath)
        unzip(fpath,fname)

def readfile(path):
    f = open(path,'r')
    s = f.readlines()

    topics = []


    
    
    finalwords = []
    for line in s:
        line = line.lower().strip().decode(errors="ignore")
        line = re.split('[-_\.:/ \"\'(),.;?\[\]!@#$%*“”‘’><{}~^&\t\\+=\\\\|]+', line)
        for word in line:
            if not word in english_punctuations and not word in wordEngStop and word != "" and word.isalpha():
                finalwords.append(word)

    # mtext = re.split('[-_:/ \"\'(),;?\[\]!@#$%*“”‘’><{}~^&\t\\+=\\\\|]+', mtext)

    # while "" in mtext:
    #     mtext.remove("")
    # print(mtext)
    # print(topics)
    #print finalwords
    return finalwords,topics

def haha1():
    # xxxx = 0
    all_words = {}
    opath = os.listdir('reuters/test')
    for ff in opath:
        simpath = os.path.join('reuters/test',ff)
        mcontent,_ = readfile(simpath)
        for word in mcontent:
            if word not in all_words.keys():
                all_words[word] = True
    pp = os.path.join('data',"test.json")
    print(pp)
    with open(pp,"w") as fp:
        json.dump(all_words, fp)

def haha2():
    # xxxx = 0
    all_words = {}
    opath = os.listdir('reuters/training')
    for ff in opath:
        simpath = os.path.join('reuters/training',ff)
        mcontent,_ = readfile(simpath)
        for word in mcontent:
            if word not in all_words.keys():
                all_words[word] = True
    pp = os.path.join('data',"training.json")
    print(pp)
    with open(pp,"w") as fp:
        json.dump(all_words, fp)

def findwords():
    #lnums = [(i*1000,(i+1)*1000) for i in range(15,21)]+[(14826,15000),(21000,21576)]    #test
    lnums = [(i*1000,(i+1)*1000) for i in range(0,14)]+[(14000,14818)]
    print(lnums)
    #lnums = [(0,1)]
    #tpath = r'E:\RCV1\words'
    tpath = os.path.join(PATH,"data")
    p = Pool(30)
    results = []
    for i in range(len(lnums)):
        start,end = lnums[i]
        print("process{0} start. Range({1},{2})".format(i,start,end))
        results.append(p.apply_async(haha,args=(start,end,tpath)))
        print("process{0} end".format(i))
    p.close()
    p.join()
    for r in results:
        print(r.get())

def isnumber(str):
    if str.count('.') == 1:
        left = str.split('.')[0]
        right = str.split('.')[1]
        lright = ''
        if str.count('-') == 1 and str[0] == '-':
            lright = left.split('-')[1]
        elif str.count('-') == 0:
            lright = left
        else:
            return False
        if right.isdigit() and lright.isdigit():
            return True
        else:
            return False
    elif str.count('.') == 0:
        if str[0] == "-":
            str2 = str[1:]
        else:
            str2 = str
        if str2.isdigit():
            return True
        return False
    else:
        return False

def allwords():
    tpath = os.path.join(PATH,"data")
    words = {}
    ind = 0
    flist = os.listdir(tpath)
    flist.sort()
    for f in flist:
        ppath = os.path.join(tpath,f)
        with open(ppath, "r") as f1:
            simjson = json.load(f1)
            for i in simjson.keys():
                if i not in words.keys():
                    words[i] = ind
                    ind += 1
    print(len(list(words.keys())))
    #print("1190" in words)
    #893198
    lens = len(list(words.keys()))
    #print(list(words.keys()))
    #assert  lens == 364830
    wembeddingwords = np.random.uniform(-1.0, 1.0, (lens, 50))
    word2vec_model = gensim.models.Word2Vec.load(r'/home/penghao/lj/Google_w2v/wiki.en.text.model')
    xx = 0
    for key in words.keys():
        # if isnumber(key):
        #     xx += 1
        if key in word2vec_model:
            #print(key)
            xx += 1
            index = words[key]
            wembeddingwords[index, :] = word2vec_model[key]
    print(xx)
    with open(os.path.join(PATH,r"words.json"), "w") as f:
        json.dump(words, f)
    f = h5py.File(os.path.join(PATH,"matrix_rcv1.h5"), "w")
    f.create_dataset("data", data=wembeddingwords)
    f.close()

def classpro():
    tpath = r'/home/user/LJ/own/RCV1/topic_codes.txt'
    haha = {}
    with open(tpath,"r") as f:
        lines = f.readlines()
        print(len(lines))
        for index,line in enumerate(lines[2:]):
            if line != '\n' and '\t' in line:
                haha[line.strip().split('\t')[0]] = index
        for k,v in haha.items():
            print(k,v)
    print(len(list(haha.keys())))
    with open(r'/home/user/LJ/own/RCV1/classes.json','w') as f:
        json.dump(haha,f)


if __name__ == "__main__":
    findwords()
    haha1()
    haha2()
    allwords()
    classpro()
