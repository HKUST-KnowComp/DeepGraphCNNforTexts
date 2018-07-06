
# 111

import numpy as np
import copy
import os

import graphcnn_option



# Global constants describing the data set.
HEIGHT = 1
WIDTH = 320 # 64*5
NUM_CHANNELS = 50
NUM_CLASSES = 24 # ????????????????????
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 49423 #
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5485 #

TRAIN_BATCH_SIZE = 128 #128 # Number of images to process in a batch.
EVAL_BATCH_SIZE = 1280  #8 # 100 # Number of images to process in a batch.

ORI_DATA_NAME = graphcnn_option.ORI_DATA_NAME
ORI_TRAIN_DATA_NAME = graphcnn_option.ORI_TRAIN_DATA_NAME
ORI_TEST_DATA_NAME = graphcnn_option.ORI_TEST_DATA_NAME
ORI_DATA_VEC_NAME = graphcnn_option.ORI_DATA_VEC_NAME
ORI_DATA_OPTION_NAME = graphcnn_option.ORI_DATA_OPTION_NAME

class DataSet(object):
    def __init__(self,data,labels,vectors,shuffled = True, index_mode = False, one_hot = True, label_used = True):
        """ Construct a DataSet.

        note: labels is 2D
        """

        self._num_examples = data.shape[0]
        self._shuffled = shuffled
        self._index_mode = index_mode
        self._one_hot = one_hot
        self._label_used = label_used
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Shuffle the data
        if shuffled:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = data[perm]
            if label_used:
                self._labels = labels[perm]
        else:
            self._data = data
            if label_used:
                self._labels = labels

        if index_mode:
            self._vectors = vectors
        else:
            data = np.zeros([self._num_examples, HEIGHT, WIDTH, NUM_CHANNELS],
                            dtype=np.float32)
            if label_used:
                labels = np.zeros([self._num_examples, NUM_CLASSES], dtype=np.int32)

            for j in range(0,self._num_examples):
                graph = np.zeros([1, self._data.shape[1], vectors.shape[1]], dtype=np.float32)
                for i in range(0, self._data.shape[1]):
                    n = self._data[j,i]
                    if n >= 0:
                        graph[0, i] = vectors[n]
                data[j] = graph
                if label_used:
                    labels[j][self._labels[j]] = 1

            self._data = data
            if label_used:
                self._labels = labels

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffled:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self._data[perm]
                if self._label_used:
                    self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        batch_data = self._data[start:end]
        if self._label_used:
            batch_labels = self._labels[start:end]

        if self._index_mode:
            data = np.zeros([batch_size, HEIGHT, WIDTH, NUM_CHANNELS],
                            dtype=np.float32)
            if self._label_used:
                labels = np.zeros([batch_size, NUM_CLASSES], dtype=np.int32)

            for j in range(0,batch_size):
                graph = np.zeros([1, batch_data.shape[1], self._vectors.shape[1]], dtype=np.float32)
                for i in range(0, batch_data.shape[1]):
                    n = batch_data[j,i]
                    if n >= 0:
                        graph[0, i] = self._vectors[n]
                data[j] = graph
                if self._label_used:
                    labels[j][batch_labels[j]] = 1
            batch_data = data
            if self._label_used:
                batch_labels = labels

        if self._label_used:
            if not self._one_hot:
                batch_labels = np.argmax(batch_labels,axis=1)

        if self._label_used:
            return batch_data,batch_labels
        else:
            return batch_data


def generate_nodelist_for_unconnectted_graph(hot,graph_size,neighborsize):
    ''' unconnectted_graph: no edge
    '''

    nodes = np.size(hot)
    tmpgraph = np.zeros([graph_size * neighborsize],dtype=np.int32)
    counter = 0
    tmphot = copy.deepcopy(hot)
    while counter < nodes and counter < np.size(tmpgraph):
        n = np.argmax(tmphot)
        tmphot[n] = -1
        tmpgraph[counter] = n+1  ###
        counter = counter + 1

    return tmpgraph


def generate_nodelist_for_undirected_graph(edges,hot,graph_size,neighborsize):
    """
    order: degree,hot
    neighbor order: edge_weight, hot

    Args:
        edges: 2D of [n,n], co-occurrence frequency
        hot: 1D of [n]
    Return:
        record: 1 * graph_size*neighbor_size * dim

    """

    if np.max(edges) < 1: # no edge
        return generate_nodelist_for_unconnectted_graph(hot,graph_size,neighborsize)

    mask = edges >= 1
    adj = mask.astype(int)   # adjacency matrix
    degree = np.sum(adj,axis=0,dtype=np.int32)  # in degreee
    nodes = np.size(hot)

    tmpgraph = np.zeros([graph_size],dtype=np.int32)
    counter = 0

    while counter < nodes and counter < graph_size:
        maxedge = np.max(degree)
        index = np.where(degree == maxedge)[0]
        degree[index] = -1
        tmphot = copy.deepcopy(hot[index])
        for _ in range(0,np.size(tmphot)):
            n = np.argmax(tmphot)
            tmphot[n] = -1
            tmpgraph[counter] = index[n]+1  ###
            counter = counter + 1
            if counter == graph_size:
                break
    assert np.size(tmpgraph) == graph_size, 'graph_size is error'

    newgraph = []
    for i in range(0,graph_size):
        n = tmpgraph[i]
        neighbors = np.zeros([neighborsize],dtype=np.int32)
        neighborsCounter = 0
        if n > 0:
            # find neighbors by (co-occurrence frequency, hot)
            neighbors[neighborsCounter] = n
            neighborsCounter = neighborsCounter + 1
            neighborPop = 0;
            while neighborsCounter < neighborsize:
                if neighborPop == neighborsCounter:
                    break
                n = neighbors[neighborPop]
                neighborPop = neighborPop + 1
                neighbor_1 = copy.deepcopy(edges[:,n-1])
                maxedge = np.max(neighbor_1)
                while maxedge > 0:
                    index = np.where(neighbor_1 == maxedge)[0]  # the return  is a tuple
                    neighbor_1[index] = -1
                    tmphot = copy.deepcopy(hot[index])
                    for _ in range(0,np.size(tmphot)):
                        if neighborsCounter >= neighborsize:
                            break
                        n = np.argmax(tmphot)
                        tmphot[n]=-1
                        m = index[n] + 1
                        if m in neighbors :
                            pass
                        else:
                            neighbors[neighborsCounter] = m
                            neighborsCounter = neighborsCounter+1
                    if neighborsCounter >= neighborsize:
                        break
                    maxedge = np.max(neighbor_1)

        newgraph = np.concatenate([newgraph, neighbors], axis=0)

    assert np.size(newgraph) == (graph_size * neighborsize), 'newgraph is error';

    return newgraph


def generate_nodelist_for_directed_graph(edges,hot,graph_size,neighborsize):
    """
    order: in_degree,hot
    neighbor order: in_edge_weight, hot

    Args:
        edges: 2D of [n,n], co-occurrence frequency
        vectors: 2D of [n,dim], node vectors
        hot: 1D of [n]
    Return:
        record: 1 * graph_size*neighbor_size * dim

    """

    if np.max(edges) < 1: # no edge
        return generate_nodelist_for_unconnectted_graph(hot,graph_size,neighborsize)

    mask = edges >= 1
    adj = mask.astype(int)   # adjacency matrix
    degree = np.sum(adj,axis=0,dtype=np.int32)  # in degreee
    nodes = np.size(hot)

    tmpgraph = np.zeros([graph_size],dtype=np.int32)
    counter = 0

    while counter < nodes and counter < graph_size:
        maxedge = np.max(degree)
        index = np.where(degree == maxedge)[0]
        degree[index] = -1
        tmphot = copy.deepcopy(hot[index])
        for _ in range(0,np.size(tmphot)):
            n = np.argmax(tmphot)
            tmphot[n] = -1
            tmpgraph[counter] = index[n]+1  ###
            counter = counter + 1
            if counter == graph_size:
                break
    assert np.size(tmpgraph) == graph_size, 'graph_size is error'

    newgraph = []
    for i in range(0,graph_size):
        tmpgraph_i = tmpgraph[i]
        neighbors = np.zeros([neighborsize],dtype=np.int32)
        neighborsCounter = 0
        if tmpgraph_i > 0:
            # find neighbors
            neighbors[neighborsCounter] = tmpgraph_i
            neighborsCounter = neighborsCounter + 1
            neighborPop = 0;
            while neighborsCounter < neighborsize:
                if neighborPop == neighborsCounter:
                    break
                neighbors_n = neighbors[neighborPop]
                neighborPop = neighborPop + 1

                neighbor_1 = copy.deepcopy(edges[:,neighbors_n-1])  # in_degree
                maxedge = np.max(neighbor_1)
                while maxedge > 0:
                    index = np.where(neighbor_1 == maxedge)[0]  # the return  is a tuple
                    neighbor_1[index] = -1
                    tmphot = copy.deepcopy(hot[index])
                    for _ in range(0,np.size(tmphot)):
                        if neighborsCounter >= neighborsize:
                            break
                        n = np.argmax(tmphot)
                        tmphot[n]=-1
                        m = index[n] + 1
                        if m in neighbors :
                            pass
                        else:
                            neighbors[neighborsCounter] = m
                            neighborsCounter = neighborsCounter+1
                    if neighborsCounter >= neighborsize:
                        break
                    maxedge = np.max(neighbor_1)

                if neighborsCounter >= neighborsize:
                    break

                neighbor_2 = copy.deepcopy(edges[neighbors_n-1,:])  # out_degree
                maxedge = np.max(neighbor_2)
                while maxedge > 0:
                    index = np.where(neighbor_2 == maxedge)[0]  # the return  is a tuple
                    neighbor_2[index] = -1
                    tmphot = copy.deepcopy(hot[index])
                    for _ in range(0, np.size(tmphot)):
                        if neighborsCounter >= neighborsize:
                            break
                        n = np.argmax(tmphot)
                        tmphot[n] = -1
                        m = index[n] + 1
                        if m in neighbors:
                            pass
                        else:
                            neighbors[neighborsCounter] = m
                            neighborsCounter = neighborsCounter + 1
                    if neighborsCounter >= neighborsize:
                        break
                    maxedge = np.max(neighbor_2)

        newgraph = np.concatenate([newgraph, neighbors], axis=0)

    assert np.size(newgraph) == (graph_size * neighborsize), 'newgraph is error';

    newgraph = newgraph.astype(np.int32)

    return newgraph


def generate_nodelist(edges,hot,graph_size,neighborsize,isDirected=True):
    if isDirected:
        return generate_nodelist_for_directed_graph(edges, hot, graph_size, neighborsize)
    else:
        return generate_nodelist_for_undirected_graph(edges, hot, graph_size, neighborsize)


def generate_dataset(data_dir, graphs, graph_size, neighborsize,shuffled = True, ont_hot = True, index_mode = False, label_used = True):
    ''' generate the data set

    index_mode: set 'index_mode' to 'True' for reduce memory footprint

    '''

    graphs_number = len(graphs)
    data = np.zeros([graphs_number,graph_size*neighborsize],dtype=np.int32)
    if label_used:
        labels = [[] for i in range(0,graphs_number)]
    for j in range(0,graphs_number):
        graph_j = graphs[j]
        nodes_weight = graph_j['nodes_weight']
        edges_weight = graph_j['edges_weight']
        nodelist_array = generate_nodelist(edges=edges_weight,hot=nodes_weight,
                                 graph_size=graph_size,neighborsize= neighborsize)
        nodes_index = graph_j['nodes_index']
        for i in range(0, np.size(nodelist_array)):
            n = nodelist_array[i]
            if n > 0:
                data[j, i] = nodes_index[n - 1]
            else:
                data[j, i] = -1

        if label_used:
            labels[j] = graph_j['label_list']  # label_list is a list

    if label_used:
        labels = np.array(labels)

    # load nodes vectors
    filename = os.path.join(data_dir, ORI_DATA_VEC_NAME)
    graphs_vectors = np.loadtxt(filename)

    if not label_used:
        labels = None

    return  DataSet(data,labels,graphs_vectors,shuffled=shuffled,index_mode=index_mode,one_hot=ont_hot,label_used=label_used)


def generate_train_data(data_dir = graphcnn_option.DATA_PATH, shuffled=True, ont_hot = True, index_mode = False):
    """ get train data

    """
    global HEIGHT, WIDTH, NUM_CHANNELS, NUM_CLASSES, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # read the option: 8
    # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
    # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
    filename = os.path.join(graphcnn_option.DATA_PATH, graphcnn_option.DATA_OPTION_NAME)
    if os.path.exists(filename):
        option = np.loadtxt(filename,dtype=np.int32)
        neighbor_size = option[0]
        graph_size = option[1]
        TRAIN_BATCH_SIZE = option[2]
        EVAL_BATCH_SIZE = option[3]
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = option[4]
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = option[5]
        NUM_CHANNELS = option[6]
        NUM_CLASSES = option[7]
        HEIGHT = 1
        WIDTH = graph_size * neighbor_size
    else:
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(32):'))
        TRAIN_BATCH_SIZE = int(input('please input train batch size:(128)'))
        EVAL_BATCH_SIZE = int(input('please input eval batch size:'))
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = int(input('please input train data size:'))
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(input('please input eval data size:'))
        NUM_CHANNELS = int(input('please input the vector dimension(50):'))
        NUM_CLASSES = int(input('please input the number of total data classes:'))
        HEIGHT = 1
        WIDTH = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % TRAIN_BATCH_SIZE, file=fr)
        print('%d' % EVAL_BATCH_SIZE, file=fr)
        print('%d' % NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, file=fr)
        print('%d' % NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, file=fr)
        print('%d' % NUM_CHANNELS, file=fr)
        print('%d' % NUM_CLASSES, file=fr)
        fr.close()

    filename = os.path.join(data_dir, graphcnn_option.TRAIN_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    graphs = []
    index = 0
    nodes_weight = []
    nodes_index = []
    edges_weight = []
    label_list = []
    for line in graphlines:
        line = line.strip()   # remove the '\n',' ' on the head and end
        linelist = line.split(' ')
        index_mod = index % 4
        if index_mod == 0: # node weight
            if len(line) == 0:  # prevent a blank line !!!!!!!
                break
            linelist = [float(i) for i in linelist]
            nodes_weight = np.array(linelist, dtype=np.float32)
        elif index_mod == 1: # node index
            linelist = [int(i) for i in linelist]
            nodes_index = np.array(linelist, dtype=np.int32)
        elif index_mod == 2: # edge weight: a[0][1]= x
            if len(line) > 0: # prevent a blank line !!!!!!!
                linelist = [float(i) for i in linelist]
                nodes_size = np.size(nodes_weight)
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                i = 0
                while i < len(linelist):
                    edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                    # edges_weight[int(linelist[i + 1])][int(linelist[i])] = linelist[i + 2]
                    i = i + 3
            else:
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
        else: # label list
            label_list = [int(i) for i in linelist]
            # label_list = np.array(linelist, dtype=np.int32)
            graphs.append({'nodes_weight': nodes_weight,
                           'nodes_index': nodes_index,
                           'edges_weight': edges_weight,
                           'label_list': label_list })
        index = index + 1

    graphs_number = len(graphs)
    assert graphs_number==NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, 'NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN is error:%d is not %d'%(graphs_number,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    print('generating train data...')
    return generate_dataset(data_dir,graphs,graph_size,neighbor_size,shuffled=shuffled,ont_hot=ont_hot,index_mode=index_mode)


def generate_eval_data(data_dir = graphcnn_option.DATA_PATH,shuffled=False, ont_hot = True,index_mode = False,label_used=True):
    """ get eval data

    """
    global HEIGHT, WIDTH, NUM_CHANNELS, NUM_CLASSES, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # read the option: 8
    # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
    # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
    filename = os.path.join(graphcnn_option.DATA_PATH, graphcnn_option.DATA_OPTION_NAME)
    if os.path.exists(filename):
        option = np.loadtxt(filename,dtype=np.int32)
        neighbor_size = option[0]
        graph_size = option[1]
        TRAIN_BATCH_SIZE = option[2]
        EVAL_BATCH_SIZE = option[3]
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = option[4]
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = option[5]
        NUM_CHANNELS = option[6]
        NUM_CLASSES = option[7]
        HEIGHT = 1
        WIDTH = graph_size * neighbor_size
    else:
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(32):'))
        TRAIN_BATCH_SIZE = int(input('please input train batch size:(128)'))
        EVAL_BATCH_SIZE = int(input('please input eval batch size:'))
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = int(input('please input train data size:'))
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(input('please input eval data size:'))
        NUM_CHANNELS = int(input('please input the vector dimension(50):'))
        NUM_CLASSES = int(input('please input the number of total data classes:'))
        HEIGHT = 1
        WIDTH = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % TRAIN_BATCH_SIZE, file=fr)
        print('%d' % EVAL_BATCH_SIZE, file=fr)
        print('%d' % NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, file=fr)
        print('%d' % NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, file=fr)
        print('%d' % NUM_CHANNELS, file=fr)
        print('%d' % NUM_CLASSES, file=fr)
        fr.close()

    filename = os.path.join(data_dir, graphcnn_option.TEST_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    graphs = []
    index = 0
    nodes_weight = []
    nodes_index = []
    edges_weight = []
    label_list = []
    for line in graphlines:
        line = line.strip()   # remove the '\n',' ' on the head and end
        linelist = line.split(' ')
        index_mod = index % 4
        if index_mod == 0: # node weight
            if len(line) == 0:  # prevent a blank line !!!!!!!
                break
            linelist = [float(i) for i in linelist]
            nodes_weight = np.array(linelist, dtype=np.float32)
        elif index_mod == 1: # node index
            linelist = [int(i) for i in linelist]
            nodes_index = np.array(linelist, dtype=np.int32)
        elif index_mod == 2: # edge weight: a[0][1]= x
            if len(line) > 0: # prevent a blank line !!!!!!!
                linelist = [float(i) for i in linelist]
                nodes_size = np.size(nodes_weight)
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
                i = 0
                while i < len(linelist):
                    edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                    # edges_weight[int(linelist[i + 1])][int(linelist[i])] = linelist[i + 2]
                    i = i + 3
            else:
                edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
        else: # one-hot label
            label_list = [int(i) for i in linelist]
            # label_one_hot = np.array(linelist, dtype=np.int32)
            graphs.append({'nodes_weight': nodes_weight,
                           'nodes_index': nodes_index,
                           'edges_weight': edges_weight,
                           'label_list': label_list })
        index = index + 1

    graphs_number = len(graphs)
    assert graphs_number==NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, 'NUM_EXAMPLES_PER_EPOCH_FOR_EVAL is error'

    print('generating test data...')
    return generate_dataset(data_dir, graphs, graph_size, neighbor_size,
                            shuffled=shuffled,ont_hot=ont_hot,index_mode=index_mode,label_used=label_used)


def generate_hier_eval_data(test_index_array, data_dir = graphcnn_option.DATA_PATH,shuffled=False,ont_hot = True,index_mode = False,label_used=True):
    """ get eval data

    """
    global HEIGHT, WIDTH, NUM_CHANNELS, NUM_CLASSES, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # read the option: 8
    # neighbor_size, graph_size, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
    # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, NUM_CHANNELS, NUM_CLASSES.
    filename = os.path.join(graphcnn_option.DATA_PATH, graphcnn_option.DATA_OPTION_NAME)
    if os.path.exists(filename):
        option = np.loadtxt(filename,dtype=np.int32)
        neighbor_size = option[0]
        graph_size = option[1]
        TRAIN_BATCH_SIZE = option[2]
        EVAL_BATCH_SIZE = option[3]
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = option[4]
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = option[5]
        NUM_CHANNELS = option[6]
        NUM_CLASSES = option[7]
        HEIGHT = 1
        WIDTH = graph_size * neighbor_size
    else:
        fr = open(filename, 'w')
        neighbor_size = int(input('please input the neighbor size(5):'))
        graph_size = int(input('please input the graph size(32):'))
        TRAIN_BATCH_SIZE = int(input('please input train batch size:(128)'))
        EVAL_BATCH_SIZE = int(input('please input eval batch size:'))
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = int(input('please input train data size:'))
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(input('please input eval data size:'))
        NUM_CHANNELS = int(input('please input the vector dimension(50):'))
        NUM_CLASSES = int(input('please input the number of total data classes:'))
        HEIGHT = 1
        WIDTH = graph_size * neighbor_size
        print('%d' % neighbor_size, file=fr)
        print('%d' % graph_size, file=fr)
        print('%d' % TRAIN_BATCH_SIZE, file=fr)
        print('%d' % EVAL_BATCH_SIZE, file=fr)
        print('%d' % NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, file=fr)
        print('%d' % NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, file=fr)
        print('%d' % NUM_CHANNELS, file=fr)
        print('%d' % NUM_CLASSES, file=fr)
        fr.close()

    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = np.size(test_index_array, axis=0)
    print('the eval graphs\' number is: %d'%NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
    # EVAL_BATCH_SIZE = int(input('please input the eval batch size:'))
    # EVAL_BATCH_SIZE = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    EVAL_BATCH_SIZE = 1982

    filename = os.path.join(data_dir, graphcnn_option.TEST_DATA_NAME)
    fr = open(filename)
    graphlines = fr.readlines()
    fr.close()
    graphs = []
    for indice in range(0, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
        index = test_index_array[indice]
        test_line_index = index * 4
        line = graphlines[test_line_index]  # node weight
        line = line.strip()   # remove the '\n',' ' on the head and end
        linelist = line.split(' ')
        if len(line) == 0:  # prevent a blank line !!!!!!!
            break
        linelist = [float(i) for i in linelist]
        nodes_weight = np.array(linelist, dtype=np.float32)

        line = graphlines[test_line_index+1] # node index
        line = line.strip()  # remove the '\n',' ' on the head and end
        linelist = line.split(' ')
        linelist = [int(i) for i in linelist]
        nodes_index = np.array(linelist, dtype=np.int32)

        line = graphlines[test_line_index+2]  # edge weight: a[0][1]= x
        line = line.strip()  # remove the '\n',' ' on the head and end
        if len(line) > 0: # prevent a blank line !!!!!!!
            linelist = line.split(' ')
            linelist = [float(i) for i in linelist]
            nodes_size = np.size(nodes_weight)
            edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)
            i = 0
            while i < len(linelist):
                edges_weight[int(linelist[i])][int(linelist[i + 1])] = linelist[i + 2]
                # edges_weight[int(linelist[i + 1])][int(linelist[i])] = linelist[i + 2]
                i = i + 3
        else:
            edges_weight = np.zeros([nodes_size, nodes_size], dtype=np.float32)

        line = graphlines[test_line_index+3]  # one-hot label
        line = line.strip()  # remove the '\n',' ' on the head and end
        linelist = line.split(' ')
        label_list = [int(i) for i in linelist]
        # label_one_hot = np.array(linelist, dtype=np.int32)
        graphs.append({'nodes_weight': nodes_weight,
                       'nodes_index': nodes_index,
                       'edges_weight': edges_weight,
                       'label_list': label_list })

    graphs_number = len(graphs)
    assert graphs_number==NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, 'NUM_EXAMPLES_PER_EPOCH_FOR_EVAL is error'

    print('generating test data...')
    return generate_dataset(data_dir, graphs, graph_size, neighbor_size,
                            shuffled=shuffled,ont_hot=ont_hot,index_mode=index_mode,label_used=label_used)










