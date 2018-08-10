import zipfile
import os

path = "ReutersCorpusVolume1/Data/ReutersCorpusVolume1_Original/CD1/"
list = os.listdir(path)

for z in list:
    file_path = os.path.join(path,z)
    zipf = zipfile.ZipFile(file_path)
    zipf.extractall('xml2')
    zipf.close()

path = "ReutersCorpusVolume1/Data/ReutersCorpusVolume1_Original/CD2/"
list = os.listdir(path)

for z in list:
    file_path = os.path.join(path,z)
    zipf = zipfile.ZipFile(file_path)
    zipf.extractall('xml2')
    zipf.close()