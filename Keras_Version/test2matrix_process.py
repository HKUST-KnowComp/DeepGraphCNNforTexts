# -*- coding: utf-8 -*-
import os
import nltk
import string
import re
import os
from nltk.corpus import wordnet as wn
import sys
import collections
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import gensim
import codecs
import h5py
import json

PATH = os.path.dirname(os.path.realpath(__file__))
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '”', '“', '’', "‘",
                        "'", '"']
wordEngStop = nltk.corpus.stopwords.words('english')
st = LancasterStemmer()


def count_words(s):
    global english_punctuations, wordEngStop, st
    tokenstr = []
    result = {}
    for line in s:
        line = line.lower().strip().decode(errors="ignore")
        line = re.split('[-_\.:/ \"\'(),.;?\[\]!@#$%*“”‘’><{}~^&\t\\+=\\\\|]+', line)
        for word in line:
            if not word in english_punctuations and not word in wordEngStop and word != "":
                orig_stem = word
                tokenstr.append(orig_stem)
                result[orig_stem] = result.get(orig_stem, 0) + 1
 
    # sort
    result = collections.OrderedDict(sorted(result.items(), key=lambda x: (x[1], x[0]), reverse=True))
    wordslist = result.keys()
    assert len(set(tokenstr)) == len(wordslist)
    # sort by count and sequence
    return (wordslist, tokenstr)


# dfs fill
def fill_table(TD_list, related_tables,target_width, qqueue):
    TD_list[0] = qqueue[0]
    count = 1
    # adjacent list
    while qqueue != [] and count < target_width:
        use_index = qqueue[0]
        del qqueue[0]
        use_list = related_tables[use_index]
        len1 = len(use_list)
        len2 = target_width - count
        if len1 >= len2:
            TD_list[count:] = use_list[:len2]
            assert len(TD_list) == target_width
            count = target_width
            break
        else:
            TD_list[count:count + len1] = use_list
            assert len(TD_list) == target_width
            count += len1
            for next_id in use_list:
                qqueue.append(next_id)
    for i in range(count, target_width):
        TD_list[i] = -1


def test_text2matrix(_str, sliding_win=3, target_width=5):
    (wordslist, tokenwords) = count_words(_str)
    wlist = list(wordslist)
    wordslist_length = len(wlist)
    if target_width > wordslist_length:
        raise ValueError("The width of matrix is larger than the total number of words in text.")
    # frequency count
    AM_table = [[0 for i in range(wordslist_length)] for j in range(wordslist_length)]
    for num in range(0, len(tokenwords) - sliding_win + 1):
        AM_table[wlist.index(tokenwords[num])][wlist.index(tokenwords[num + 1])] += 1
        AM_table[wlist.index(tokenwords[num])][wlist.index(tokenwords[num + 2])] += 1
        AM_table[wlist.index(tokenwords[num + 1])][wlist.index(tokenwords[num + 2])] += 1
        AM_table[wlist.index(tokenwords[num + 1])][wlist.index(tokenwords[num])] += 1
        AM_table[wlist.index(tokenwords[num + 2])][wlist.index(tokenwords[num])] += 1
        AM_table[wlist.index(tokenwords[num + 2])][wlist.index(tokenwords[num + 1])] += 1
    # related table: descending order
    related_tables = {}
    for i in range(wordslist_length):
        related_tables[i] = [[index, num] for index, num in enumerate(AM_table[i]) if num > 0 and index != i]
        related_tables[i].sort(key=lambda x: x[1], reverse=True)
        related_tables[i] = [element[0] for element in related_tables[i]]
    TD_table = [[0 for i in range(target_width)] for j in range(wordslist_length)]
    
    for i in range(wordslist_length):
        fill_table(TD_table[i], related_tables,target_width, [i])

    return wordslist, TD_table


def matrix_vector(wordslist, TD_table, target_width, word_vector_size):
    wlist = list(wordslist)
    word2vec_model = gensim.models.Word2Vec.load('english.bin')
    TTD_table = np.zeros((word_vector_size, len(wlist), target_width), dtype=np.float32)

    with open(r'./words_index.json', "r") as f3:
        w_idnex = json.load(f3)

    h5 = h5py.File(r'./matrix.h5', 'r')
    wdata = h5['data'].value

    for num_i in range(len(wlist)):
        for num_j in range(target_width):
            if TD_table[num_i][num_j] > -1:
                try:
                    aword = wlist[TD_table[num_i][num_j]]
                    wind = w_idnex[aword]
                    c_wordvector = wdata[wind]
                    # c_wordvector = word2vec_model[wlist[TD_table[num_i][num_j]]]
                    # TTD_table[:, num_i, num_j] = c_wordvector
                except:
                    # c_wordvector = np.zeros((word_vector_size), dtype=np.float32)
                    aword = wlist[TD_table[num_i][num_j]]
                    wind = w_idnex[aword]
                    c_wordvector = wdata[wind]
            else:
                c_wordvector = np.zeros((word_vector_size), dtype=np.float32)
            TTD_table[:, num_i, num_j] = c_wordvector
    return (TTD_table)


def process(path, dirsim,slise_window, target_width, word_vector_size, words_limit, catrgory, class_nums):
    ipath = os.path.join(path, dirsim)
    sec_lst = os.listdir(ipath)
    _X = None
    _y = None
    flag = 0

    one_hot_codes = np.eye(class_nums)

    for sec_dir in sec_lst:
        tfpath = os.path.join(ipath, sec_dir)
        #print(tfpath)
        with open(tfpath, "rb") as fff:
            a = fff.readlines()
            (wordslist, TD_table) = test_text2matrix(a, slise_window, target_width)
            TTD_table = matrix_vector(wordslist, TD_table, target_width, word_vector_size)
            shape0, shape1, shape2 = TTD_table.shape
            #print(shape0, shape1, shape2)
            final_one_TTD = None
            if shape1 < words_limit:
                final_one_TTD = np.zeros((shape0, words_limit, shape2), dtype=np.float32)
                final_one_TTD[:, :shape1, :shape2] = TTD_table
            else:
                final_one_TTD = TTD_table[:, :words_limit, :shape2]
            #                 print(final_one_TTD.shape)
            #                 print(final_one_TTD[:,20,4])
            final_one_TTD = final_one_TTD.reshape((1, word_vector_size, words_limit, target_width))
            # print(final_one_TTD.shape)
            _yxx = one_hot_codes[catrgory]
            _yxx = _yxx.reshape(1,-1)
            # print(_yxx.shape)
            if flag == 0:
                _X = final_one_TTD
                _y = _yxx
                flag = 1
            else:
                _X = np.concatenate((_X, final_one_TTD), axis=0)
                _y = np.concatenate((_y, _yxx), axis=0)
            # print("----" * 20)
    # print(catrgory+1,_y)
    # print(_X.shape)
    # print(_y.shape)
    fpath = r"./data/c%d.h5" % (catrgory+1)
    f = h5py.File(fpath, "w")
    f.create_dataset("datax", data=_X)
    f.create_dataset("datay", data=_y)
    f.close()


if __name__ == '__main__':
    start,end = int(sys.argv[1]),int(sys.argv[2])
    print(start,end)
    # sliding window for graph-of-words
    slise_window = 3
    # width in word neighboring
    target_width = 10
    # Word vector dimension
    word_vector_size = 50
    # word number in graph-of-words
    words_limit = 300
    # labels for classification task
    class_nums = 20
    path = r'./20news-19997/20_newsgroups'
    #path = os.path.join(PATH, "20news-19997/20_newsgroups")

    dir_lst = os.listdir(path)
    for i, dirsim in enumerate(dir_lst[start:end]):
        catrgory = start+i
        process(path,dirsim, slise_window, target_width, word_vector_size, words_limit,catrgory,class_nums)
    print('Done!!!')
