import numpy as np
import os
import math
import graphcnn_option

DATA_PATH = graphcnn_option.DATA_PATH   # Path to data directory
TRAIN_DATA_DIR = graphcnn_option.TRAIN_DATA_DIR

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


# the random extract ratio for train
# SPILT_RATIO = 0.9 # for single label
# SPILT_RATIO = 0.7 # 0.5 #for multi label

ORI_DATA_NAME = graphcnn_option.ORI_DATA_NAME
ORI_TRAIN_DATA_NAME = graphcnn_option.ORI_TRAIN_DATA_NAME
ORI_TEST_DATA_NAME = graphcnn_option.ORI_TEST_DATA_NAME
ORI_DATA_VEC_NAME = graphcnn_option.ORI_DATA_VEC_NAME
ORI_DATA_OPTION_NAME = graphcnn_option.ORI_DATA_OPTION_NAME


def generate_data_with_keynode_and_undirected_graph(data_dir = DATA_PATH):
    """ get input data and data info

    node has an attribute: node_is_key
    edge is undirected

    """

    data_option = np.loadtxt(os.path.join(data_dir, ORI_DATA_OPTION_NAME),dtype=np.int32)
    # data_option_graphsNumber = data_option[0]
    # data_option_labelNumber = data_option[1]
    # data_option_vectorDim = data_option[2]

    filename = os.path.join(data_dir, ORI_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    graphs = []  # total graphs set
    graphsNode = [] # node number in each graph
    graphsKeyNode = [] # keynode number in each graph
    graphsLabel = [] # label number in each graph
    graphsDegree = [] # node degree in each graph
    index = 0
    nodes_size = 0
    nodes_weight = []
    nodes_iskey = []
    nodes_index = []
    edges_weight = []
    label_list = []
    for line in graphlines:
        line = line.strip()   # remove the '\n',' ' on the head and end
        # if len(line) == 0: # prevent a blank line !!!!!!!
        #     break
        index_mod = index % 6
        if index_mod == 0: # node size
            if len(line) == 0: # prevent last blank line !!!!!!!
                break
            linelist = line.split(' ')
            nodes_size = int(linelist[0])
            graphsNode.append(nodes_size)
        elif index_mod == 1: # node weight
            nodes_weight = line
        elif index_mod == 2:  # node_iskey
            linelist = line.split(' ')
            linelist = [int(i) for i in linelist]
            nodes_iskey = np.array(linelist, dtype=np.int32)
            graphsKeyNode.append(np.sum(nodes_iskey))
        elif index_mod == 3: # node index
            nodes_index = line
        elif index_mod == 4: # edge weight: a[0][1]= x
            if len(line) > 0: # prevent a blank line !!!!!!!
                linelist = line.split(' ')
                linelist = [float(i) for i in linelist]
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                i = 0
                while i < len(linelist):
                    edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                    i = i + 3
            else:
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
            # degree
            graphsDegree.extend(list(np.sum((edges_weight >= 1).astype(int), axis=0, dtype=np.int32)))
            edges_weight = line
        else: # label list for single label or multi-label
            linelist = line.split(' ')
            linelist = [int(i) for i in linelist]
            label_list = np.array(linelist, dtype=np.int32)
            graphsLabel.append(np.size(label_list))
            graphs.append({'nodes_weight': nodes_weight,
                           'nodes_index': nodes_index,
                           'edges_weight': edges_weight,
                           'label_list_vector': label_list,
                           'label_list': line})
        index = index + 1

    # load nodes vectors
    dimension = data_option[2] #
    graphs_number = len(graphs) #
    assert graphs_number==data_option[0], 'graphs_number != data_option[0]'
    graphs_class = data_option[1] #
    labelsGraph = np.zeros([graphs_class],dtype=np.int32) #
    for i in range(0, graphs_number):
        labelsGraph[graphs[i]['label_list_vector']] +=1

    # data info:
    graphsNode_max = np.max(graphsNode)
    graphsNode_min = np.min(graphsNode)
    graphsNode_median = np.median(graphsNode)
    graphsNode_mean = np.mean(graphsNode)
    graphsKeyNode_max = np.max(graphsKeyNode)
    graphsKeyNode_min = np.min(graphsKeyNode)
    graphsKeyNode_median = np.median(graphsKeyNode)
    graphsKeyNode_mean = np.mean(graphsKeyNode)
    graphsDegree_max = np.max(graphsDegree)
    graphsDegree_min = np.min(graphsDegree)
    graphsDegree_median = np.median(graphsDegree)
    graphsDegree_mean = np.mean(graphsDegree)
    graphsLabel_max = np.max(graphsLabel)
    graphsLabel_min = np.min(graphsLabel)
    graphsLabel_median = np.median(graphsLabel)
    graphsLabel_mean = np.mean(graphsLabel)
    labelsGraph_max = np.max(labelsGraph)
    labelsGraph_min = np.min(labelsGraph)
    labelsGraph_median = np.median(labelsGraph)
    labelsGraph_mean = np.mean(labelsGraph)


    graph_size1 = np.max([graphsKeyNode_max, graphsNode_median]); ######
    graph_size = np.power(2, math.ceil(np.log2(graph_size1)));

    print('data info: (you may see the dataInfo.txt for more details)')
    filename = os.path.join(data_dir, 'dataInfo.txt')
    fr = open(filename,mode='w')
    print('data_dir:%s' %data_dir,file =fr)
    print('data info:',file=fr)
    np.set_printoptions(threshold=np.nan)
    print('total graphs number:%d'%graphs_number)
    print('total classes number:%d'%graphs_class)
    print('total graphs number:%d' % graphs_number , file=fr)
    print('total classes number:%d' % graphs_class , file=fr)
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median))
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median))
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median),file=fr)
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median),file=fr)
    print(' ')
    print(' ',file=fr)
    print('graphsNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median))
    print('graphsKeyNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsKeyNode_max, graphsKeyNode_min, graphsKeyNode_mean, graphsKeyNode_median))
    print('graphsDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsDegree_max, graphsDegree_min, graphsDegree_mean, graphsDegree_median))
    print('graph_size:%d -> %d' %(graph_size1,graph_size))
    print('dimension:%d' % dimension)

    print('graphNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median), file= fr)
    print('graphsKeyNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsKeyNode_max, graphsKeyNode_min, graphsKeyNode_mean, graphsKeyNode_median) , file= fr)
    print('graphDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsDegree_max, graphsDegree_min, graphsDegree_mean, graphsDegree_median) , file= fr)
    print('graph_size:%d -> %d' % (graph_size1, graph_size), file= fr)
    print('dimension:%d' % dimension, file= fr)

    # split data:
    SPILT_RATIO = float(input('please input the split ratio for train data:'))

    samples_per_label = []
    for i in range(0,graphs_class):
        samples_per_label.append([])
    for i in range(0, graphs_number):
        label_list_vector = graphs[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label[j].append(i)
    print('\nsamples number per label(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d' % (i,len(samples_per_label[i])),file=fr)

    train_index = np.zeros([graphs_number],dtype=np.int32)
    for i in range(0,graphs_class):
        label_len = len(samples_per_label[i])
        perm = np.arange(0,label_len)
        np.random.shuffle(perm)
        train_len = int(label_len * SPILT_RATIO)
        for j in range(0,train_len):
            train_index[samples_per_label[i][perm[j]]] = 1
    graphs_for_train = []
    graphs_for_test = []
    for i in range(0, graphs_number):
        if train_index[i] == 1: # for train
            graphs_for_train.append(graphs[i])
        else: # for test
            graphs_for_test.append(graphs[i])

    print('\n\n------------------split data-----------------',file=fr)
    # train
    samples_per_label_for_train = []
    graphs_number_for_train = len(graphs_for_train)
    for i in range(0,graphs_class):
        samples_per_label_for_train.append([])
    for i in range(0, graphs_number_for_train):
        label_list_vector = graphs_for_train[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_train[j].append(i)
    print('samples number for train:%d(%.4f)' % (graphs_number_for_train, graphs_number_for_train/graphs_number), file=fr)
    print('samples number per label for train(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_train[i]),len(samples_per_label_for_train[i])/len(samples_per_label[i])),file=fr)
    # test
    samples_per_label_for_test = []
    graphs_number_for_test = len(graphs_for_test)
    for i in range(0,graphs_class):
        samples_per_label_for_test.append([])
    for i in range(0, graphs_number_for_test):
        label_list_vector = graphs_for_test[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_test[j].append(i)
    print('\nsamples number for test:%d(%.4f)' % (graphs_number_for_test, graphs_number_for_test/graphs_number), file=fr)
    print('samples number per label for test(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_test[i]),len(samples_per_label_for_test[i])/len(samples_per_label[i])),file=fr)

    fr.close()
    str = input('next will generate train data and test data, enter to continue(y/n):')
    if str == 'y' or str == 'Y':
        if graphs_number_for_train > 0:
            print('generating train data...')
            filename = os.path.join(data_dir, 'data.train')
            fr = open(filename,'w')
            for i in range(0,graphs_number_for_train):
                print(graphs_for_train[i]['nodes_weight'],file=fr)
                print(graphs_for_train[i]['nodes_index'], file=fr)
                print(graphs_for_train[i]['edges_weight'], file=fr)
                print(graphs_for_train[i]['label_list'], file=fr)
            fr.close()
        if graphs_number_for_test > 0:
            print('generating test data...')
            filename = os.path.join(data_dir, 'data.test')
            fr = open(filename, 'w')
            for i in range(0, graphs_number_for_test):
                print(graphs_for_test[i]['nodes_weight'], file=fr)
                print(graphs_for_test[i]['nodes_index'], file=fr)
                print(graphs_for_test[i]['edges_weight'], file=fr)
                print(graphs_for_test[i]['label_list'], file=fr)
            fr.close()
    elif str == 'n' or str == 'N':
        return
    else:
        print('invalid input!')
        return

    str = input('whether to generate the data option(y/n):')
    if str == 'y' or str == 'Y':
        # generate the option: 8
        # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
        # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
        filename = os.path.join(DATA_PATH, 'data.option')
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(32):'))
        train_batch_size = int(input('please input train batch size:(128)'))
        eval_batch_size = int(input('please input eval batch size:'))
        num_examples_per_epoch_for_train = graphs_number_for_train
        num_examples_per_epoch_for_eval = graphs_number_for_test
        num_channels = dimension
        num_classes = graphs_class
        height = 1
        width = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % train_batch_size, file=fr)
        print('%d' % eval_batch_size, file=fr)
        print('%d' % num_examples_per_epoch_for_train, file=fr)
        print('%d' % num_examples_per_epoch_for_eval, file=fr)
        print('%d' % num_channels, file=fr)
        print('%d' % num_classes, file=fr)
        fr.close()
    return

def generate_data_with_directed_graph(data_dir = DATA_PATH):
    """ get input data and data info

    node has no such attribute: node_is_key.
    edge is directed

    """

    data_option = np.loadtxt(os.path.join(data_dir, ORI_DATA_OPTION_NAME), dtype=np.int32)
    # data_option_graphsNumber = data_option[0]
    # data_option_labelNumber = data_option[1]
    # data_option_vectorDim = data_option[2]

    filename = os.path.join(data_dir, ORI_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    graphs = []
    graphsNode = []
    graphsLabel = []
    graphsInDegree = []
    index = 0
    nodes_size = 0
    nodes_weight = []
    nodes_index = []
    edges_weight = []
    label_list = []
    for line in graphlines:
        line = line.strip()   # remove the '\n',' ' on the head and end
        # if len(line) == 0: # prevent a blank line !!!!!!!
        #     break
        index_mod = index % 5
        if index_mod == 0: # node size
            if len(line) == 0: # prevent last blank line !!!!!!!
                break
            linelist = line.split(' ')
            nodes_size = int(linelist[0])
            graphsNode.append(nodes_size)
        elif index_mod == 1: # node weight
            nodes_weight = line
        elif index_mod == 2: # node index
            nodes_index = line
        elif index_mod == 3: # edge weight: a[0][1]= x
            if len(line) > 0: # prevent a blank line !!!!!!!
                linelist = line.split(' ')
                linelist = [float(i) for i in linelist]
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                i = 0
                while i < len(linelist):
                    edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                    i = i + 3
            else:
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
            # inDegree
            graphsInDegree.extend(list(np.sum((edges_weight >= 1).astype(int), axis=0, dtype=np.int32)))
            edges_weight = line
        else: # one-hot label
            linelist = line.split(' ')
            linelist = [int(i) for i in linelist]
            label_list = np.array(linelist, dtype=np.int32)
            graphsLabel.append(np.size(label_list))
            graphs.append({'nodes_weight': nodes_weight,
                           'nodes_index': nodes_index,
                           'edges_weight': edges_weight,
                           'label_list_vector': label_list,
                           'label_list': line})
        index = index + 1

    # load nodes vectors
    dimension = data_option[2]  #
    graphs_number = len(graphs)  #
    assert graphs_number == data_option[0], 'graphs_number != data_option[0]'
    graphs_class = data_option[1]  #
    labelsGraph = np.zeros([graphs_class], dtype=np.int32)  #
    for i in range(0, graphs_number):
        labelsGraph[graphs[i]['label_list_vector']] += 1

    # data info:
    graphsNode_max = np.max(graphsNode)
    graphsNode_min = np.min(graphsNode)
    graphsNode_median = np.median(graphsNode)
    graphsNode_mean = np.mean(graphsNode)
    graphsInDegree_max = np.max(graphsInDegree)
    graphsInDegree_min = np.min(graphsInDegree)
    graphsInDegree_median = np.median(graphsInDegree)
    graphsInDegree_mean = np.mean(graphsInDegree)
    graphsLabel_max = np.max(graphsLabel)
    graphsLabel_min = np.min(graphsLabel)
    graphsLabel_median = np.median(graphsLabel)
    graphsLabel_mean = np.mean(graphsLabel)
    labelsGraph_max = np.max(labelsGraph)
    labelsGraph_min = np.min(labelsGraph)
    labelsGraph_median = np.median(labelsGraph)
    labelsGraph_mean = np.mean(labelsGraph)


    graph_size1 = graphsNode_median; ######
    graph_size = np.power(2, math.ceil(np.log2(graph_size1)));

    print('data info: (you may see the dataInfo.txt for more details)')
    filename = os.path.join(data_dir, 'dataInfo.txt')
    fr = open(filename,mode='w')
    print('data_dir:%s' % data_dir,file=fr)
    print('data info:',file=fr)
    np.set_printoptions(threshold=np.nan)
    print('total graphs number:%d'%graphs_number)
    print('total classes number:%d'%graphs_class)
    print('total graphs number:%d' % graphs_number , file=fr)
    print('total classes number:%d' % graphs_class , file=fr)
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median))
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median))
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median),file=fr)
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median),file=fr)
    print(' ')
    print(' ',file=fr)
    print('graphsNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median))
    print('graphsInDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsInDegree_max, graphsInDegree_min, graphsInDegree_mean, graphsInDegree_median))
    print('graph_size:%d -> %d' %(graph_size1,graph_size))
    print('dimension:%d' % dimension)

    print('graphNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median), file= fr)
    print('graphInDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsInDegree_max, graphsInDegree_min, graphsInDegree_mean, graphsInDegree_median) , file= fr)
    print('graph_size:%d -> %d' % (graph_size1, graph_size), file= fr)
    print('dimension:%d' % dimension, file= fr)

    # split data:
    SPILT_RATIO = float(input('please input the split ratio for train data:'))

    samples_per_label = []
    for i in range(0, graphs_class):
        samples_per_label.append([])
    for i in range(0, graphs_number):
        label_list_vector = graphs[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label[j].append(i)
    print('\nsamples number per label(label:number):', file=fr)
    for i in range(0, graphs_class):
        print('  %d : %d' % (i, len(samples_per_label[i])), file=fr)

    train_index = np.zeros([graphs_number],dtype=np.int32)
    for i in range(0,graphs_class):
        label_len = len(samples_per_label[i])
        perm = np.arange(0,label_len)
        np.random.shuffle(perm)
        train_len = int(label_len * SPILT_RATIO)
        for j in range(0,train_len):
            train_index[samples_per_label[i][perm[j]]] = 1
    graphs_for_train = []
    graphs_for_test = []
    for i in range(0, graphs_number):
        if train_index[i] == 1: # for train
            graphs_for_train.append(graphs[i])
        else: # for test
            graphs_for_test.append(graphs[i])

    print('\n\n------------------split data-----------------',file=fr)
    # train
    samples_per_label_for_train = []
    graphs_number_for_train = len(graphs_for_train)
    for i in range(0,graphs_class):
        samples_per_label_for_train.append([])
    for i in range(0, graphs_number_for_train):
        label_list_vector = graphs_for_train[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_train[j].append(i)
    print('samples number for train:%d(%.4f)' % (graphs_number_for_train, graphs_number_for_train/graphs_number), file=fr)
    print('samples number per label for train(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_train[i]),len(samples_per_label_for_train[i])/len(samples_per_label[i])),file=fr)
    # test
    samples_per_label_for_test = []
    graphs_number_for_test = len(graphs_for_test)
    for i in range(0,graphs_class):
        samples_per_label_for_test.append([])
    for i in range(0, graphs_number_for_test):
        label_list_vector = graphs_for_test[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_test[j].append(i)
    print('\nsamples number for test:%d(%.4f)' % (graphs_number_for_test, graphs_number_for_test/graphs_number), file=fr)
    print('samples number per label for test(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_test[i]),len(samples_per_label_for_test[i])/len(samples_per_label[i])),file=fr)

    fr.close()
    str = input('next will generate train data and test data, enter to continue(y/n):')
    if str == 'y' or str == 'Y':
        if graphs_number_for_train > 0:
            print('generating train data...')
            filename = os.path.join(data_dir, 'data.train')
            fr = open(filename,'w')
            for i in range(0,graphs_number_for_train):
                print(graphs_for_train[i]['nodes_weight'],file=fr)
                print(graphs_for_train[i]['nodes_index'], file=fr)
                print(graphs_for_train[i]['edges_weight'], file=fr)
                print(graphs_for_train[i]['label_list'], file=fr)
            fr.close()
        if graphs_number_for_test > 0:
            print('generating test data...')
            filename = os.path.join(data_dir, 'data.test')
            fr = open(filename, 'w')
            for i in range(0, graphs_number_for_test):
                print(graphs_for_test[i]['nodes_weight'], file=fr)
                print(graphs_for_test[i]['nodes_index'], file=fr)
                print(graphs_for_test[i]['edges_weight'], file=fr)
                print(graphs_for_test[i]['label_list'], file=fr)
            fr.close()
    elif str == 'n' or str == 'N':
        return
    else:
        print('invalid input!')
        return

    str = input('whether to generate the data option(y/n):')
    if str == 'y' or str == 'Y':
        # generate the option: 8
        # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
        # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
        filename = os.path.join(DATA_PATH, 'data.option')
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(32):'))
        train_batch_size = int(input('please input train batch size:(128)'))
        eval_batch_size = int(input('please input eval batch size:'))
        num_examples_per_epoch_for_train = graphs_number_for_train
        num_examples_per_epoch_for_eval = graphs_number_for_test
        num_channels = dimension
        num_classes = graphs_class
        height = 1
        width = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % train_batch_size, file=fr)
        print('%d' % eval_batch_size, file=fr)
        print('%d' % num_examples_per_epoch_for_train, file=fr)
        print('%d' % num_examples_per_epoch_for_eval, file=fr)
        print('%d' % num_channels, file=fr)
        print('%d' % num_classes, file=fr)
        fr.close()
    return

def generate_data(keynode=False, isDirected=False, data_dir = DATA_PATH):
    if keynode and (not isDirected):
        return generate_data_with_keynode_and_undirected_graph(data_dir)
    else:
        return generate_data_with_directed_graph(data_dir)


def generate_data_with_separated_graph(data_dir = DATA_PATH):
    """ get input data and data info

    node has no such attribute: node_is_key.
    edge is directed

    """

    data_option = np.loadtxt(os.path.join(data_dir, ORI_DATA_OPTION_NAME), dtype=np.int32)
    # data_option_train_graphsNumber = data_option[0]
    # data_option_train_labelNumber = data_option[1]
    # data_option_vectorDim = data_option[2]
    # data_option_test_graphsNumber = data_option[3]
    # data_option_test_labelNumber = data_option[4]
    data_option_train_graphsNumber = data_option[0]
    data_option_test_graphsNumber = data_option[1]
    data_option_labelNumber = data_option[2]
    data_option_vectorDim = data_option[3]

    filename_list = [os.path.join(data_dir, ORI_TRAIN_DATA_NAME), os.path.join(data_dir, ORI_TEST_DATA_NAME)]
    graphs = []
    graphsNode = []
    graphsLabel = []
    graphsInDegree = []
    index = 0
    nodes_size = 0
    nodes_weight = []
    nodes_index = []
    edges_weight = []
    label_list = []
    badsamples = []
    for filename in filename_list:
        fr = open(filename)
        graphlines = fr.readlines()
        fr.close()
        badsamples_count = 0
        for line in graphlines:
            line = line.strip()   # remove the '\n',' ' on the head and end
            # print(index,end='@')
            # print(line)
            # if len(line) == 0: # prevent a blank line !!!!!!!
            #     break
            index_mod = index % 5
            if index_mod == 0: # node size
                if len(line) == 0: # prevent last blank line !!!!!!!
                    nodes_size = 0
                else:
                    linelist = line.split(' ')
                    nodes_size = int(linelist[0])
                graphsNode.append(nodes_size)
            elif index_mod == 1: # node weight
                nodes_weight = line
            elif index_mod == 2: # node index
                nodes_index = line
            elif index_mod == 3: # edge weight: a[0][1]= x
                if len(line) > 0: # prevent a blank line !!!!!!!
                    linelist = line.split(' ')
                    linelist = [float(i) for i in linelist]
                    edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                    i = 0
                    while i < len(linelist):
                        edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                        i = i + 3
                else:
                    edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                # inDegree
                graphsInDegree.extend(list(np.sum((edges_weight >= 1).astype(int), axis=0, dtype=np.int32)))
                edges_weight = line
            else: # one-hot label
                if line=='null' or len(line)==0 or nodes_size==0 or len(nodes_weight)==0 or len(nodes_index)==0:
                    badsamples_count += 1
                else:
                    linelist = line.split(' ')
                    linelist = [int(i) for i in linelist]
                    label_list = np.array(linelist, dtype=np.int32)
                    graphsLabel.append(np.size(label_list))
                    graphs.append({'nodes_weight': nodes_weight,
                                   'nodes_index': nodes_index,
                                   'edges_weight': edges_weight,
                                   'label_list_vector': label_list,
                                   'label_list': line})
            index = index + 1
        badsamples.append(badsamples_count)

    # load nodes vectors
    dimension = data_option_vectorDim  #
    graphs_number = len(graphs)  #

    # print(graphs_number)
    # print(data_option_train_graphsNumber)
    # print(data_option_test_graphsNumber)
    # print(badsamples[0])
    # print(badsamples[1])

    data_option_train_graphsNumber = data_option_train_graphsNumber - badsamples[0]
    data_option_test_graphsNumber = data_option_test_graphsNumber - badsamples[1]

    assert graphs_number == data_option_test_graphsNumber+data_option_train_graphsNumber, 'graphs_number is error'
    # if data_option_train_labelNumber > data_option_test_labelNumber:
    #     graphs_class = data_option_train_labelNumber  #
    # else:
    #     graphs_class = data_option_test_labelNumber
    graphs_class = data_option_labelNumber
    labelsGraph = np.zeros([graphs_class], dtype=np.int32)  #

    for i in range(0, graphs_number):
        labelsGraph[graphs[i]['label_list_vector']] += 1

    # data info:
    graphsNode_max = np.max(graphsNode)
    graphsNode_min = np.min(graphsNode)
    graphsNode_median = np.median(graphsNode)
    graphsNode_mean = np.mean(graphsNode)
    graphsInDegree_max = np.max(graphsInDegree)
    graphsInDegree_min = np.min(graphsInDegree)
    graphsInDegree_median = np.median(graphsInDegree)
    graphsInDegree_mean = np.mean(graphsInDegree)
    graphsLabel_max = np.max(graphsLabel)
    graphsLabel_min = np.min(graphsLabel)
    graphsLabel_median = np.median(graphsLabel)
    graphsLabel_mean = np.mean(graphsLabel)
    labelsGraph_max = np.max(labelsGraph)
    labelsGraph_min = np.min(labelsGraph)
    labelsGraph_median = np.median(labelsGraph)
    labelsGraph_mean = np.mean(labelsGraph)


    graph_size1 = graphsNode_median; ######
    graph_size = np.power(2, math.ceil(np.log2(graph_size1)));

    print('data info: (you may see the dataInfo.txt for more details)')
    filename = os.path.join(data_dir, 'dataInfo.txt')
    fr = open(filename,mode='w')
    print('data_dir:%s' % data_dir,file=fr)
    print('data info:',file=fr)
    np.set_printoptions(threshold=np.nan)
    print('total graphs number:%d(%d/%d)(%d/%d)' % (
    graphs_number, data_option_train_graphsNumber, data_option_test_graphsNumber,badsamples[0],badsamples[1]))
    print('total classes number:%d'%graphs_class)
    print('total graphs number:%d' % graphs_number , file=fr)
    print('total classes number:%d' % graphs_class , file=fr)
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median))
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median))
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median),file=fr)
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median),file=fr)
    print(' ')
    print(' ',file=fr)
    print('graphsNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median))
    print('graphsInDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsInDegree_max, graphsInDegree_min, graphsInDegree_mean, graphsInDegree_median))
    print('graph_size:%d -> %d' %(graph_size1,graph_size))
    print('dimension:%d' % dimension)

    print('graphNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median), file= fr)
    print('graphInDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsInDegree_max, graphsInDegree_min, graphsInDegree_mean, graphsInDegree_median) , file= fr)
    print('graph_size:%d -> %d' % (graph_size1, graph_size), file= fr)
    print('dimension:%d' % dimension, file= fr)


    # split data:

    samples_per_label = []
    for i in range(0, graphs_class):
        samples_per_label.append([])
    for i in range(0, graphs_number):
        label_list_vector = graphs[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label[j].append(i)
    print('\nsamples number per label(label:number):', file=fr)
    for i in range(0, graphs_class):
        print('  %d : %d' % (i, len(samples_per_label[i])), file=fr)

    graphs_for_train = []
    graphs_for_test = []
    for i in range(0, data_option_train_graphsNumber):
        graphs_for_train.append(graphs[i])
    for i in range(data_option_train_graphsNumber, graphs_number):
        graphs_for_test.append(graphs[i])

    print('\n\n------------------split data-----------------',file=fr)
    # train
    samples_per_label_for_train = []
    graphs_number_for_train = len(graphs_for_train)
    for i in range(0,graphs_class):
        samples_per_label_for_train.append([])
    for i in range(0, graphs_number_for_train):
        label_list_vector = graphs_for_train[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_train[j].append(i)
    print('samples number for train:%d(%.4f)' % (graphs_number_for_train, graphs_number_for_train/graphs_number), file=fr)
    print('samples number per label for train(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_train[i]),len(samples_per_label_for_train[i])/len(samples_per_label[i])),file=fr)
    # test
    samples_per_label_for_test = []
    graphs_number_for_test = len(graphs_for_test)
    for i in range(0,graphs_class):
        samples_per_label_for_test.append([])
    for i in range(0, graphs_number_for_test):
        label_list_vector = graphs_for_test[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_test[j].append(i)
    print('\nsamples number for test:%d(%.4f)' % (graphs_number_for_test, graphs_number_for_test/graphs_number), file=fr)
    print('samples number per label for test(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_test[i]),len(samples_per_label_for_test[i])/len(samples_per_label[i])),file=fr)

    fr.close()
    str = input('next will generate train data and test data, enter to continue(y/n):')
    if str == 'y' or str == 'Y':
        if graphs_number_for_train > 0:
            print('generating train data...')
            filename = os.path.join(data_dir, graphcnn_option.TRAIN_DATA_NAME)
            fr = open(filename,'w')
            for i in range(0,graphs_number_for_train):
                print(graphs_for_train[i]['nodes_weight'],file=fr)
                print(graphs_for_train[i]['nodes_index'], file=fr)
                print(graphs_for_train[i]['edges_weight'], file=fr)
                print(graphs_for_train[i]['label_list'], file=fr)
            fr.close()
        if graphs_number_for_test > 0:
            print('generating test data...')
            filename = os.path.join(data_dir, graphcnn_option.TEST_DATA_NAME)
            fr = open(filename, 'w')
            for i in range(0, graphs_number_for_test):
                print(graphs_for_test[i]['nodes_weight'], file=fr)
                print(graphs_for_test[i]['nodes_index'], file=fr)
                print(graphs_for_test[i]['edges_weight'], file=fr)
                print(graphs_for_test[i]['label_list'], file=fr)
            fr.close()
    elif str == 'n' or str == 'N':
        return
    else:
        print('invalid input!')
        return

    str = input('whether to generate the data option(y/n):')
    if str == 'y' or str == 'Y':
        # generate the option: 8
        # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
        # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
        filename = os.path.join(DATA_PATH, graphcnn_option.DATA_OPTION_NAME)
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(32):'))
        train_batch_size = int(input('please input train batch size:(128)'))
        eval_batch_size = int(input('please input eval batch size:'))
        num_examples_per_epoch_for_train = graphs_number_for_train
        num_examples_per_epoch_for_eval = graphs_number_for_test
        num_channels = dimension
        num_classes = graphs_class
        height = 1
        width = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % train_batch_size, file=fr)
        print('%d' % eval_batch_size, file=fr)
        print('%d' % num_examples_per_epoch_for_train, file=fr)
        print('%d' % num_examples_per_epoch_for_eval, file=fr)
        print('%d' % num_channels, file=fr)
        print('%d' % num_classes, file=fr)
        fr.close()
    return

def generate_hier_data_with_separated_graph(data_dir = DATA_PATH):
    """ get input data and data info

    node has no such attribute: node_is_key.
    edge is directed

    """

    data_option = np.loadtxt(os.path.join(data_dir, ORI_DATA_OPTION_NAME), dtype=np.int32)
    # data_option_train_graphsNumber = data_option[0]
    # data_option_train_labelNumber = data_option[1]
    # data_option_vectorDim = data_option[2]
    # data_option_test_graphsNumber = data_option[3]
    # data_option_test_labelNumber = data_option[4]
    data_option_train_graphsNumber = data_option[0]
    data_option_test_graphsNumber = data_option[1]
    data_option_labelNumber = data_option[2]
    data_option_vectorDim = data_option[3]

    filename = os.path.join(data_dir, graphcnn_option.HIER_DIR_NAME,
                            graphcnn_option.HIER_labels_remap_file)
    labels_remap = np.loadtxt(filename, dtype=int)
    data_option_labelNumber = np.size(labels_remap,axis=0)

    graphs = []
    graphsNode = []
    graphsLabel = []
    graphsInDegree = []
    nodes_size = 0
    nodes_weight = []
    nodes_index = []
    edges_weight = []
    # train data
    filename = os.path.join(data_dir, ORI_TRAIN_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    filename = os.path.join(data_dir, graphcnn_option.HIER_DIR_NAME,
                            graphcnn_option.HIER_train_graphs_index_file)
    train_index_array = np.loadtxt(filename,dtype=int)
    data_option_train_graphsNumber = np.size(train_index_array,axis=0)
    filename = os.path.join(data_dir, graphcnn_option.HIER_DIR_NAME,
                            graphcnn_option.HIER_train_labels_file)
    fr = open(filename)
    label_lines = fr.readlines()
    fr.close()
    for indice in range(0,data_option_train_graphsNumber):
        index = train_index_array[indice]
        train_line_index = index * 5
        line = graphlines[train_line_index] # nodes_size
        line = line.strip()  # remove the '\n',' ' on the head and end
        if len(line) == 0:  # prevent last blank line !!!!!!!
            break
        linelist = line.split(' ')
        nodes_size = int(linelist[0])
        graphsNode.append(nodes_size)
        nodes_weight = (graphlines[train_line_index+1]).strip()  # node weight
        nodes_index = (graphlines[train_line_index+2]).strip()  # node index
        line = graphlines[train_line_index+3]  # edge weight: a[0][1]= x
        line = line.strip()
        if len(line) > 0:  # prevent a blank line !!!!!!!
            linelist = line.split(' ')
            linelist = [float(i) for i in linelist]
            edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
            i = 0
            while i < len(linelist):
                edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                i = i + 3
        else:
            edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
        # inDegree
        graphsInDegree.extend(list(np.sum((edges_weight >= 1).astype(int), axis=0, dtype=np.int32)))
        edges_weight = line

        # label:
        line = label_lines[indice]
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(i) for i in linelist]
        label_list = np.array(linelist, dtype=np.int32)
        graphsLabel.append(np.size(label_list))
        graphs.append({'nodes_weight': nodes_weight,
                       'nodes_index': nodes_index,
                       'edges_weight': edges_weight,
                       'label_list_vector': label_list,
                       'label_list': line})

    # test data
    filename = os.path.join(data_dir, ORI_TEST_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    if graphcnn_option.HIER_test_used:
        filename = os.path.join(data_dir, graphcnn_option.HIER_DIR_NAME,
                                graphcnn_option.HIER_test_graphs_index_file)
        test_index_array = np.loadtxt(filename, dtype=int)
        data_option_test_graphsNumber = np.size(test_index_array, axis=0)
        filename = os.path.join(data_dir, graphcnn_option.HIER_DIR_NAME,
                                graphcnn_option.HIER_test_labels_file)
        fr = open(filename)
        label_lines = fr.readlines()
        fr.close()
        for indice in range(0, data_option_test_graphsNumber):
            index = test_index_array[indice]
            test_line_index = index * 5
            line = graphlines[test_line_index]  # nodes_size
            line = line.strip()  # remove the '\n',' ' on the head and end
            if len(line) == 0:  # prevent last blank line !!!!!!!
                break
            linelist = line.split(' ')
            nodes_size = int(linelist[0])
            graphsNode.append(nodes_size)
            nodes_weight = (graphlines[test_line_index + 1]).strip()  # node weight
            nodes_index = (graphlines[test_line_index + 2]).strip()  # node index
            line = graphlines[test_line_index + 3]  # edge weight: a[0][1]= x
            line = line.strip()
            if len(line) > 0:  # prevent a blank line !!!!!!!
                linelist = line.split(' ')
                linelist = [float(i) for i in linelist]
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                i = 0
                while i < len(linelist):
                    edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                    i = i + 3
            else:
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
            # inDegree
            graphsInDegree.extend(list(np.sum((edges_weight >= 1).astype(int), axis=0, dtype=np.int32)))
            edges_weight = line

            # label:
            line = label_lines[indice]
            line = line.strip()
            linelist = line.split(' ')
            linelist = [int(i) for i in linelist]
            label_list = np.array(linelist, dtype=np.int32)
            graphsLabel.append(np.size(label_list))
            graphs.append({'nodes_weight': nodes_weight,
                           'nodes_index': nodes_index,
                           'edges_weight': edges_weight,
                           'label_list_vector': label_list,
                           'label_list': line})
    else:
        index = 0
        for line in graphlines:
            line = line.strip()  # remove the '\n',' ' on the head and end
            # if len(line) == 0: # prevent a blank line !!!!!!!
            #     break
            index_mod = index % 5
            if index_mod == 0:  # node size
                if len(line) == 0:  # prevent last blank line !!!!!!!
                    break
                linelist = line.split(' ')
                nodes_size = int(linelist[0])
                graphsNode.append(nodes_size)
            elif index_mod == 1:  # node weight
                nodes_weight = line
            elif index_mod == 2:  # node index
                nodes_index = line
            elif index_mod == 3:  # edge weight: a[0][1]= x
                if len(line) > 0:  # prevent a blank line !!!!!!!
                    linelist = line.split(' ')
                    linelist = [float(i) for i in linelist]
                    edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                    i = 0
                    while i < len(linelist):
                        edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                        i = i + 3
                else:
                    edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                # inDegree
                graphsInDegree.extend(list(np.sum((edges_weight >= 1).astype(int), axis=0, dtype=np.int32)))
                edges_weight = line
            else:  # one-hot label
                linelist = line.split(' ')
                linelist = [int(i) for i in linelist]
                label_list = np.array(linelist, dtype=np.int32)
                graphsLabel.append(np.size(label_list))
                graphs.append({'nodes_weight': nodes_weight,
                               'nodes_index': nodes_index,
                               'edges_weight': edges_weight,
                               'label_list_vector': label_list,
                               'label_list': line})
            index = index + 1

    # load nodes vectors
    dimension = data_option_vectorDim  #
    graphs_number = len(graphs)  #
    assert graphs_number == data_option_test_graphsNumber+data_option_train_graphsNumber, 'graphs_number is error'
    # if data_option_train_labelNumber > data_option_test_labelNumber:
    #     graphs_class = data_option_train_labelNumber  #
    # else:
    #     graphs_class = data_option_test_labelNumber
    graphs_class = data_option_labelNumber
    labelsGraph = np.zeros([graphs_class], dtype=np.int32)  #

    for i in range(0, graphs_number):
        labelsGraph[graphs[i]['label_list_vector']] += 1

    # data info:
    graphsNode_max = np.max(graphsNode)
    graphsNode_min = np.min(graphsNode)
    graphsNode_median = np.median(graphsNode)
    graphsNode_mean = np.mean(graphsNode)
    graphsInDegree_max = np.max(graphsInDegree)
    graphsInDegree_min = np.min(graphsInDegree)
    graphsInDegree_median = np.median(graphsInDegree)
    graphsInDegree_mean = np.mean(graphsInDegree)
    graphsLabel_max = np.max(graphsLabel)
    graphsLabel_min = np.min(graphsLabel)
    graphsLabel_median = np.median(graphsLabel)
    graphsLabel_mean = np.mean(graphsLabel)
    labelsGraph_max = np.max(labelsGraph)
    labelsGraph_min = np.min(labelsGraph)
    labelsGraph_median = np.median(labelsGraph)
    labelsGraph_mean = np.mean(labelsGraph)


    graph_size1 = graphsNode_median; ######
    graph_size = np.power(2, math.ceil(np.log2(graph_size1)));

    print('Hier'+ graphcnn_option.rootstr +'data info: (you may see the dataInfo.txt for more details)')
    filename = os.path.join(data_dir, 'hier'+ graphcnn_option.rootstr + 'dataInfo.txt')
    fr = open(filename,mode='w')
    print('data_dir:%s' % data_dir,file=fr)
    print('data info:',file=fr)
    np.set_printoptions(threshold=np.nan)
    print('total graphs number:%d(%d/%d)'%(graphs_number,data_option_train_graphsNumber,data_option_test_graphsNumber))
    print('total classes number:%d'%graphs_class)
    print('total graphs number:%d(%d/%d)'%(graphs_number,data_option_train_graphsNumber,data_option_test_graphsNumber) , file=fr)
    print('total classes number:%d' % graphs_class , file=fr)
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median))
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median))
    print('class info for graphs: max:%d, min:%d, mean:%d, median:%d' % (
        graphsLabel_max, graphsLabel_min, graphsLabel_mean, graphsLabel_median),file=fr)
    print('graph info for classes: max:%d, min:%d, mean:%d, median:%d' % (
        labelsGraph_max, labelsGraph_min, labelsGraph_mean, labelsGraph_median),file=fr)
    print(' ')
    print(' ',file=fr)
    print('graphsNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median))
    print('graphsInDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsInDegree_max, graphsInDegree_min, graphsInDegree_mean, graphsInDegree_median))
    print('graph_size:%d -> %d' %(graph_size1,graph_size))
    print('dimension:%d' % dimension)

    print('graphNode: max:%d, min:%d, mean:%d, median:%d' % (
        graphsNode_max, graphsNode_min, graphsNode_mean, graphsNode_median), file= fr)
    print('graphInDegree: max:%d, min:%d, mean:%d, median:%d' % (
        graphsInDegree_max, graphsInDegree_min, graphsInDegree_mean, graphsInDegree_median) , file= fr)
    print('graph_size:%d -> %d' % (graph_size1, graph_size), file= fr)
    print('dimension:%d' % dimension, file= fr)

    # split data:

    samples_per_label = []
    for i in range(0, graphs_class):
        samples_per_label.append([])
    for i in range(0, graphs_number):
        label_list_vector = graphs[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label[j].append(i)
    print('\nsamples number per label(label:number):', file=fr)
    for i in range(0, graphs_class):
        print('  %d : %d' % (i, len(samples_per_label[i])), file=fr)

    graphs_for_train = []
    graphs_for_test = []
    for i in range(0, data_option_train_graphsNumber):
        graphs_for_train.append(graphs[i])
    for i in range(data_option_train_graphsNumber, graphs_number):
        graphs_for_test.append(graphs[i])

    print('\n\n------------------split data-----------------',file=fr)
    # train
    samples_per_label_for_train = []
    graphs_number_for_train = len(graphs_for_train)
    for i in range(0,graphs_class):
        samples_per_label_for_train.append([])
    for i in range(0, graphs_number_for_train):
        label_list_vector = graphs_for_train[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_train[j].append(i)
    print('samples number for train:%d(%.4f)' % (graphs_number_for_train, graphs_number_for_train/graphs_number), file=fr)
    print('samples number per label for train(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_train[i]),len(samples_per_label_for_train[i])/len(samples_per_label[i])),file=fr)
    # test
    samples_per_label_for_test = []
    graphs_number_for_test = len(graphs_for_test)
    for i in range(0,graphs_class):
        samples_per_label_for_test.append([])
    for i in range(0, graphs_number_for_test):
        label_list_vector = graphs_for_test[i]['label_list_vector']
        for j in label_list_vector:
            samples_per_label_for_test[j].append(i)
    print('\nsamples number for test:%d(%.4f)' % (graphs_number_for_test, graphs_number_for_test/graphs_number), file=fr)
    print('samples number per label for test(label:number):',file=fr)
    for i in range(0,graphs_class):
        print('  %d : %d (%.4f)' % (i,len(samples_per_label_for_test[i]),len(samples_per_label_for_test[i])/len(samples_per_label[i])),file=fr)

    fr.close()
    str = input('next will generate train data and test data, enter to continue(y/n):')
    if str == 'y' or str == 'Y':
        if graphs_number_for_train > 0:
            print('generating train data...')
            filename = os.path.join(data_dir, graphcnn_option.TRAIN_DATA_NAME)
            fr = open(filename,'w')
            for i in range(0,graphs_number_for_train):
                print(graphs_for_train[i]['nodes_weight'],file=fr)
                print(graphs_for_train[i]['nodes_index'], file=fr)
                print(graphs_for_train[i]['edges_weight'], file=fr)
                print(graphs_for_train[i]['label_list'], file=fr)
            fr.close()
        if graphs_number_for_test > 0:
            print('generating test data...')
            filename = os.path.join(data_dir, graphcnn_option.TEST_DATA_NAME)
            fr = open(filename, 'w')
            for i in range(0, graphs_number_for_test):
                print(graphs_for_test[i]['nodes_weight'], file=fr)
                print(graphs_for_test[i]['nodes_index'], file=fr)
                print(graphs_for_test[i]['edges_weight'], file=fr)
                print(graphs_for_test[i]['label_list'], file=fr)
            fr.close()
    elif str == 'n' or str == 'N':
        return
    else:
        print('invalid input!')
        return

    str = input('whether to generate the data option(y/n):')
    if str == 'y' or str == 'Y':
        # generate the option: 8
        # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
        # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
        filename = os.path.join(DATA_PATH, graphcnn_option.DATA_OPTION_NAME)
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(192):'))
        train_batch_size = int(input('please input train batch size(128):'))
        eval_batch_size = int(input('please input eval batch size:'))
        num_examples_per_epoch_for_train = graphs_number_for_train
        num_examples_per_epoch_for_eval = graphs_number_for_test
        num_channels = dimension
        num_classes = graphs_class
        height = 1
        width = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % train_batch_size, file=fr)
        print('%d' % eval_batch_size, file=fr)
        print('%d' % num_examples_per_epoch_for_train, file=fr)
        print('%d' % num_examples_per_epoch_for_eval, file=fr)
        print('%d' % num_channels, file=fr)
        print('%d' % num_classes, file=fr)
        fr.close()
    return




def main(argv=None):
    # generate_data(keynode=False, isDirected=True, data_dir=TRAIN_DATA_DIR)
    # generate_data_with_separated_graph(data_dir=TRAIN_DATA_DIR)
    generate_hier_data_with_separated_graph(data_dir=TRAIN_DATA_DIR)


if __name__ == '__main__':
    main()

