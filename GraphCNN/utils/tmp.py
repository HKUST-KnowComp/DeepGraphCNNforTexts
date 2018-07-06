

import numpy as np
import os
import shutil

# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))

def xx():
    filename = 'graphcnn_hier_eval_without_labels.py'
    DIR = '.'
    pathDir =  os.listdir(DIR)
    for path in pathDir:
        if len(path)>5 and path[0:5]=='LSHTC':
            sourceFile = os.path.join(DIR, filename)
            targetFile = os.path.join(DIR,path,filename)
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)


a = np.array([[1,2,3],[1,2,3]])
a = np.reshape(a,[-1,1])
print(a)