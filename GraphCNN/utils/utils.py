import numpy as np
import  os

def main():
    filename = '/home/heyu/PycharmProjects/graphCNN/data/label_groups'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = '/home/heyu/PycharmProjects/graphCNN/data/label_groups_info'
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        print(len(linelist),file=fr)
    fr.close()

    filename = '/home/heyu/PycharmProjects/graphCNN/data/example_groups'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = '/home/heyu/PycharmProjects/graphCNN/data/example_groups_info'
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        print(len(linelist),file=fr)
    fr.close()


if __name__ == '__main__':
    main()
