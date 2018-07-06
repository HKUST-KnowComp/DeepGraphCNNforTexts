
import numpy as np
import os
import graphcnn_option


THRESHOLD1 = 1000
THRESHOLD2 = 800
THRESHOLD3 = 600
THRESHOLD4 = 400
SUM = 0
ROOT = 0  # the root of subgraph

DATA_PATH = graphcnn_option.DATA_PATH   # Path to data directory
TRAIN_DATA_DIR = graphcnn_option.TRAIN_DATA_DIR
EVAL_DATA_DIR = graphcnn_option.EVAL_DATA_DIR
HIER_DIR_NAME = graphcnn_option.HIER_DIR_NAME

def generate_labels_list_per_example(data_dir = TRAIN_DATA_DIR):
    """ get example(graph)-labels file:
    1 3 4
    6 9
    将各个graph的标签单独提取出来，每一行对应一个graph
    ...

    """
    filename = os.path.join(data_dir, 'data.train')
    fr = open(filename,'r')
    graphlines = fr.readlines()
    fr.close()
    filename = os.path.join(data_dir, 'train_labels_remap')
    fr = open(filename, 'w')
    index = 1
    for line in graphlines:
        if index % 4 == 0:
            line = line.strip()  # remove the '\n',' ' on the head and end
            print(line, file=fr)
        index = index + 1
    fr.close()

    filename = os.path.join(data_dir, 'data.test')
    fr = open(filename, 'r')
    graphlines = fr.readlines()
    fr.close()
    filename = os.path.join(data_dir, 'test_labels_remap')
    fr = open(filename, 'w')
    index = 1
    for line in graphlines:
        if index % 4 == 0:
            line = line.strip()  # remove the '\n',' ' on the head and end
            print(line, file=fr)
        index = index + 1
    fr.close()

def group_labels_by_examples(data_dir = TRAIN_DATA_DIR):
    '''According to the examples(graphs), labels can be divided into different groups.
       examples for each group are disjoint from other groups.

    Return:
        label groups list and example groups list
    '''

    label_groups_list = []
    example_groups_list = []

    # example-labels file
    filename = os.path.join(data_dir, 'train_labels_remap')
    fr = open(filename, 'r')
    example_labels_lines = fr.readlines()
    fr.close()
    examples_number = len(example_labels_lines)
    labels_number = 2297
    examples_flag = np.zeros([examples_number],dtype=int)
    for label in range(0,labels_number):
        # step1
        flag = 0
        for group in label_groups_list:
            if label in group:
                flag = 1
                break
        if flag == 1:
            continue

        # step2
        label_group = []
        example_group = []
        label_group.append(label)
        count = 0
        while count < len(label_group):
            label_count = label_group[count]
            count  += 1
            # step3
            for i in range(0,examples_number):
                if examples_flag[i] == 0:
                    line = example_labels_lines[i]
                    line = line.strip()
                    linelist = line.split(' ')
                    linelist = [int(j) for j in linelist]
                    if label_count in linelist:
                        examples_flag[i] = 1
                        example_group.append(i)
                        for j in linelist:
                            if j not in label_group:
                                label_group.append(j)
        label_groups_list.append(label_group)
        example_groups_list.append(example_group)

    filename = os.path.join(data_dir, 'label_groups')
    fr = open(filename, 'w')
    for list_i in label_groups_list:
        for i in list_i:
            print(i,end=' ',file=fr)
        print('',file=fr)
    fr.close()

    filename = os.path.join(data_dir, 'example_groups')
    fr = open(filename, 'w')
    for list_i in example_groups_list:
        for i in list_i:
            print(i, end=' ', file=fr)
        print('', file=fr)
    fr.close()

# 生成原始的样本标签
def generate_example_labels_orig(data_dir = TRAIN_DATA_DIR):

    filename = os.path.join(data_dir,'remap.txt')
    ori_remap = np.loadtxt(filename, dtype=int)
    remap2ori_dict = {}
    for i in range(0,len(ori_remap)):
        remap2ori_dict[ori_remap[i][1]] = ori_remap[i][0]

    filename = os.path.join(data_dir,'train_labels_remap')
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = os.path.join(data_dir,'train_labels_orig')
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(j) for j in linelist]
        labels_remap = []
        for one in linelist:
            labels_remap.append(remap2ori_dict[one])
        for one in labels_remap:
            print(one, end=' ', file=fr)
        print('', file=fr)
    fr.close()

    filename = os.path.join(data_dir, 'test_labels_remap')
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = os.path.join(data_dir, 'test_labels_orig')
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(j) for j in linelist]
        labels_remap = []
        for one in linelist:
            labels_remap.append(remap2ori_dict[one])
        for one in labels_remap:
            print(one, end=' ', file=fr)
        print('', file=fr)
    fr.close()

# 将hierarchy的父子关系合并，统计每个Node的所有直接父亲和孩子
def generate_hier_dict_with_parent_child():
    hier_dict = {}
    filename = '../graphCNN_data/hier_relation'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    index = 0
    for line in lines:
        index_mod = index % 3
        if index_mod == 0:
            line = line.strip()
            label = int(line)
        elif index_mod == 1:
            line = line.strip()
            if len(line) == 0:
                parent_list = []
            else:
                linelist = line.split(' ')
                parent_list = [int(j) for j in linelist]
        elif index_mod == 2:
            line = line.strip()
            if len(line) == 0:
                child_list = []
            else:
                linelist = line.split(' ')
                child_list = [int(j) for j in linelist]
            hier_dict[label] = [parent_list, child_list]
        index += 1
    return hier_dict

# 生成扩展后的样本标签
def generate_example_labels_orig_expand():
    hier_dict = generate_hier_dict_with_parent_child()
    filename = '../graphCNN_data/train_labels_orig'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = '../graphCNN_data/train_labels_orig_expand'
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(j) for j in linelist]
        count = 0
        while count < len(linelist):
            label_count = linelist[count]
            count += 1
            parent_list = hier_dict[label_count][0]
            for i in parent_list:
                if i not in linelist:
                    linelist.append(i)
        for i in linelist:
            print(i, end=' ', file=fr)
        print('', file=fr)
    fr.close()

    filename = '../graphCNN_data/test_labels_orig'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = '../graphCNN_data/test_labels_orig_expand'
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(j) for j in linelist]
        count = 0
        while count < len(linelist):
            label_count = linelist[count]
            count += 1
            parent_list = hier_dict[label_count][0]
            for i in parent_list:
                if i not in linelist:
                    linelist.append(i)
        for i in linelist:
            print(i, end=' ', file=fr)
        print('', file=fr)
    fr.close()

# 根据hierarchy生成最顶层的根节点列表（可能不唯一）
# root: 0
def generate_hier0_data():  # 0
    ''' hier0_remap,hier0_labels,hier0_graphs_index,hier0_graphs'''

    hier_dict = generate_hier_dict_with_parent_child()
    hier0_remap = []
    # find root nodes:(who has no parent)
    for k in hier_dict.keys():
        if len(hier_dict[k][0])==0:
            hier0_remap.append(k)

    hier0_remap_len = len(hier0_remap)
    filename = '../graphCNN_data/hier0_remap'
    fr = open(filename,'w')
    for i in range(0,hier0_remap_len):
        print('%d %d' % (hier0_remap[i],i), file=fr)
    fr.close()


def _subfunc1(hier_remap, root, level):
    hier_remap_len = len(hier_remap)
    filename = '../data/lshtc/hier%d_%d_remap' % (level, root)
    fr = open(filename, 'w')
    for i in range(0, hier_remap_len):
        print('%d %d' % (hier_remap[i], i), file=fr)
    fr.close()
    filename = '../data/lshtc/hier%d_%d_labels' % (level, root)
    fr_label = open(filename, 'w')
    filename = '../data/lshtc/hier%d_%d_graphs_index' % (level, root)
    fr_graph = open(filename, 'w')
    filename = '../data/lshtc/example_labels_orig_expand'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    for i in range(0, len(lines)):
        line = lines[i]
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(k) for k in linelist]
        flag = 0
        for j in range(0, len(hier_remap)):
            if hier_remap[j] in linelist:
                flag = 1
                print(j, end=' ', file=fr_label)
        if flag == 1:
            print(i, file=fr_graph)
            print('', file=fr_label)
    fr_graph.close()
    fr_label.close()

def _generate_hier_n_data_subfunc(hier_dict,hier_leafList_dict,root,level,subtrees_used):
    subtrees_used.append(root)
    root_leaf_list = hier_leafList_dict[root]
    if len(root_leaf_list) == 1 and root_leaf_list[0] == root: # leaf node
        return
    if len(root_leaf_list) < THRESHOLD1:
        _subfunc1(root_leaf_list, root, level)
        return
    hier_remap = hier_dict[root][1]
    _subfunc1(hier_remap,root,level)
    for label in hier_remap:
        if label not in subtrees_used:
            _generate_hier_n_data_subfunc(hier_dict,hier_leafList_dict,label,level+1,subtrees_used)

def generate_hier_n_data_root2leaf():  # 2143406
    '''
    hierN_XXX_remap
    hierN_XXX_labels
    hierN_XXX_graphs_index
    hierN_XXX_graphs
    '''

    filename = '../data/lshtc/hier_relation'
    fr = open(filename,'r')
    lines = fr.readlines()
    fr.close()
    hier_dict = {}
    index = 0
    for line in lines:
        index_mod = index % 3
        if index_mod == 0:
            line = line.strip()
            label = int(line)
        elif index_mod == 1:
            line = line.strip()
            if len(line)==0:
                parent_list = []
            else:
                linelist = line.split(' ')
                parent_list = [int(j) for j in linelist]
        elif index_mod == 2:
            line = line.strip()
            if len(line) == 0:
                child_list = []
            else:
                linelist = line.split(' ')
                child_list = [int(j) for j in linelist]
            hier_dict[label] = [parent_list,child_list]
        index +=1

    root = 2143406  # the root of subgraph

    # filename = '../data/lshtc/hier_relation_leafList'
    # fr = open(filename,'w')
    # print(len(_compute_leaf_node_of_sub_tree(hier_dict,root,fr)))
    # fr.close()

    filename = '../data/lshtc/hier_relation_leafList'
    fr = open(filename,'r')
    hier_leafList = fr.readlines()
    fr.close()
    hier_leafList_dict = {}
    for line in hier_leafList:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(j) for j in linelist]
        hier_leafList_dict[linelist[0]] = linelist[1:]
    subtrees_used = []
    _generate_hier_n_data_subfunc(hier_dict,hier_leafList_dict,root,1,subtrees_used)


def _subfunc1_compute_leaf_node_of_sub_tree_copy(hier_dict,root,hier_leafList_dict,allleafList):
    '''计算以root为根节点的树的叶节点列表'''
    print(root)
    if root in hier_leafList_dict.keys():
        return
    hier_remap = hier_dict[root][1]
    hier_remap_len = len(hier_remap)
    if hier_remap_len == 0:  # leaf node
        hier_leafList_dict[root]=[root]
        return

    if root in allleafList:
        leaf_root = [root]
    else:
        leaf_root = []
    for label in hier_remap:
        _subfunc1_compute_leaf_node_of_sub_tree(hier_dict,label,hier_leafList_dict,allleafList)
        leaf_list = hier_leafList_dict[label]
        for one in leaf_list:
            if one not in leaf_root:
                leaf_root.append(one)
    hier_leafList_dict[root] = leaf_root
    return

def _subfunc1_compute_leaf_node_of_sub_tree(hier_dict,root,allleafList):
    '''计算以root为根节点的树的叶节点列表'''
    leaf_root = []
    handled_list = []
    unhandled_list = [root]
    while len(unhandled_list)>0:
        label = unhandled_list[0]
        unhandled_list.remove(label)
        handled_list.append(label)
        if label in allleafList:
            leaf_root.append(label)
        hier_remap = hier_dict[label][1]
        for one in hier_remap:
            if one not in handled_list and one not in unhandled_list:
                unhandled_list.append(one)

    return leaf_root

def _subfunc1_compute_leaf_node_of_sub_tree_all(hier_dict,hier_leafList_dict):
    # gong 2318 label
    filename = os.path.join(TRAIN_DATA_DIR, 'remap.txt')
    ori_remap = np.loadtxt(filename, dtype=int)
    allleafList = ori_remap[:, 0]
    hier_leafList_dict.clear()
    for i in range(0,2318):
        hier_leafList_dict[i] = _subfunc1_compute_leaf_node_of_sub_tree(hier_dict,i,allleafList)
    return hier_leafList_dict

def _subfunc2_generate_hier_remap(hier_leafList_dict, root_list):
    hier_remap = []
    root_str = '_'
    for root in root_list:
        root_str += '%d_' % root
        root_leaf = hier_leafList_dict[root]
        for one in root_leaf:
            if one not in hier_remap:
                hier_remap.append(one)
    hier_remap_len = len(hier_remap)

    print(root_list,end=' ')
    print(hier_remap_len)

    if len(root_list) > 3:
        filename = '../graphCNN_data/hier/hier_%d_%d_others_'%(root_list[0],root_list[1])+'rootstr'
        fr = open(filename, 'w')
        print(root_str,file=fr)
        fr.close()
        root_str = '_%d_%d_others_'%(root_list[0],root_list[1])

    HIER_labels_remap_file = 'hier' + root_str + 'remap'
    HIER_train_graphs_index_file = 'hier' + root_str + 'train_graphs_index'
    HIER_train_labels_file = 'hier' + root_str + 'train_labels'
    HIER_test_graphs_index_file = 'hier' + root_str + 'test_graphs_index'
    HIER_test_labels_file = 'hier' + root_str + 'test_labels'

    filename = os.path.join(TRAIN_DATA_DIR,HIER_DIR_NAME,HIER_labels_remap_file)
    fr = open(filename, 'w')
    for i in range(0, hier_remap_len):
        print('%d %d' % (hier_remap[i], i), file=fr)
    fr.close()
    filename = os.path.join(TRAIN_DATA_DIR, HIER_DIR_NAME, HIER_train_labels_file)
    fr_label = open(filename, 'w')
    filename = os.path.join(TRAIN_DATA_DIR, HIER_DIR_NAME, HIER_train_graphs_index_file)
    fr_graph = open(filename, 'w')
    filename = os.path.join(TRAIN_DATA_DIR,'train_labels_orig_expand')
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    for i in range(0, len(lines)):
        line = lines[i]
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(k) for k in linelist]
        flag = 0
        for j in range(0, len(hier_remap)):
            if hier_remap[j] in linelist:
                flag = 1
                print(j, end=' ', file=fr_label)
        if flag == 1:
            print(i, file=fr_graph)
            print('', file=fr_label)
    fr_graph.close()
    fr_label.close()

    filename = os.path.join(TRAIN_DATA_DIR, HIER_DIR_NAME, HIER_test_labels_file)
    fr_label = open(filename, 'w')
    filename = os.path.join(TRAIN_DATA_DIR, HIER_DIR_NAME, HIER_test_graphs_index_file)
    fr_graph = open(filename, 'w')
    filename = os.path.join(TRAIN_DATA_DIR, 'test_labels_orig_expand')
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    for i in range(0, len(lines)):
        line = lines[i]
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(k) for k in linelist]
        flag = 0
        for j in range(0, len(hier_remap)):
            if hier_remap[j] in linelist:
                flag = 1
                print(j, end=' ', file=fr_label)
        if flag == 1:
            print(i, file=fr_graph)
            print('', file=fr_label)
    fr_graph.close()
    fr_label.close()

def _subfunc3_generate_hier_n_data_leaf2root(hier_dict,hier_leafList_dict,root):
    child_list = hier_dict[root][1]
    for one in child_list:
        if len(hier_leafList_dict[one]) > THRESHOLD1:
            _subfunc3_generate_hier_n_data_leaf2root(hier_dict,hier_leafList_dict,one)
            # hier_leafList_dict.clear()
            # _subfunc1_compute_leaf_node_of_sub_tree(hier_dict, ROOT, hier_leafList_dict)
    for one in child_list:
        if len(hier_leafList_dict[one]) >= THRESHOLD2:
            _subfunc2_generate_hier_remap(hier_leafList_dict,[one])
            hier_leafList_dict[one] = [one]
            one_child_list = hier_dict[one][1]
            hier_dict[one][1] = []
            for one_child in one_child_list:
                if one in hier_dict[one_child][0]:
                    hier_dict[one_child][0].remove(one)
            # hier_leafList_dict.clear()
            _subfunc1_compute_leaf_node_of_sub_tree_all(hier_dict, hier_leafList_dict)

    if len(hier_leafList_dict[root]) > THRESHOLD1:
        for one in child_list:
            if len(hier_leafList_dict[one]) >= THRESHOLD3:
                _subfunc2_generate_hier_remap(hier_leafList_dict, [one])
                hier_leafList_dict[one] = [one]
                one_child_list = hier_dict[one][1]
                hier_dict[one][1] = []
                for one_child in one_child_list:
                    if one in hier_dict[one_child][0]:
                        hier_dict[one_child][0].remove(one)
                # hier_leafList_dict.clear()
                _subfunc1_compute_leaf_node_of_sub_tree_all(hier_dict, hier_leafList_dict)

    if len(hier_leafList_dict[root]) > THRESHOLD1:
        child_not_zero = []
        for one in child_list:
            if len(hier_leafList_dict[one]) !=  1:
                child_not_zero.append(one)
        while len(child_not_zero)>0:
            root_list = []
            root_list_leaf = []
            for one in child_not_zero:
                leaf_sum = len(root_list_leaf)
                leaf_list = hier_leafList_dict[one]
                for two in leaf_list:
                    if two not in root_list_leaf:
                        leaf_sum +=1
                if leaf_sum <= THRESHOLD1:     # ?????????????????????????
                    root_list.append(one)
                    for two in leaf_list:
                        if two not in root_list_leaf:
                            root_list_leaf.append(two)
            if len(root_list_leaf) < THRESHOLD4 and len(root_list)==len(child_not_zero):# ????????
                break
            _subfunc2_generate_hier_remap(hier_leafList_dict, root_list)
            for one in root_list:
                child_not_zero.remove(one)
                hier_leafList_dict[one] = [one]
                one_child_list = hier_dict[one][1]
                hier_dict[one][1] = []
                for one_child in one_child_list:
                    if one in hier_dict[one_child][0]:
                        hier_dict[one_child][0].remove(one)
            # hier_leafList_dict.clear()
            _subfunc1_compute_leaf_node_of_sub_tree_all(hier_dict, hier_leafList_dict)
            # _subfunc1_compute_leaf_node_of_sub_tree(hier_dict, ROOT, hier_leafList_dict)
    # print(len(hier_leafList_dict[root]))



def generate_hier_n_data_leaf2root(data_dir = TRAIN_DATA_DIR):
    '''
        hier_XXX_remap
        hier_XXX_labels
        hier_XXX_graphs_index
        hier_XXX_data
    '''
    hier_dict = generate_hier_dict_with_parent_child()
    root = ROOT
    hier_leafList_dict = {}
    _subfunc1_compute_leaf_node_of_sub_tree_all(hier_dict,hier_leafList_dict)

    handled_list = []
    if len(hier_leafList_dict[root]) > THRESHOLD1:
        _subfunc3_generate_hier_n_data_leaf2root(hier_dict, hier_leafList_dict, root)
    _subfunc2_generate_hier_remap(hier_leafList_dict, [root])
    return



def find_root_node():
    filename = '../data/lshtc/remap'
    leaf = np.loadtxt(filename, dtype=int)
    leaf = leaf[:,0]
    filepath = './data'
    pathDir = os.listdir(filepath)
    sum = 0
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        example_label_array = np.loadtxt(child, dtype=int)
        label_array = example_label_array[:, 0]
        print('len:%d'%(len(label_array)))
        for one in label_array:
            if one not in leaf:
                print(one)
                sum+=1
    print('sum:%d'%sum)


def generate_eval_result_file():
    ''' train:456886,  max label:167593(11400)
        test:81262
    '''
    result_dict = {}
    filepath = './hier_result_leaf'
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.getsize(child):
            example_label_array = np.loadtxt(child, dtype=int)
            for i in range(0,np.size(example_label_array,axis=0)):
                example_index = example_label_array[i,0]
                label_index = example_label_array[i,1]
                if example_index in result_dict.keys():
                    if label_index not in result_dict[example_index]:
                        result_dict[example_index].append(label_index)
                else:
                    result_dict[example_index] = [label_index]

    example_dict = {}
    filepath = './hier_result_leaf_exp'
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.getsize(child):
            fr = open(child, 'r')
            lines = fr.readlines()
            fr.close()
            for line in lines:
                line = line.strip()
                linelist = line.split(' ')
                example = int(linelist[0])
                label = int(linelist[1])
                label_value = float(linelist[2])
                if example in example_dict.keys():
                    if label_value > example_dict[example]['label_value']:
                        example_dict[example]['label'] = label
                        example_dict[example]['label_value'] = label_value
                else:
                    example_dict[example] = {'label': label, 'label_value': label_value}

    filename = './result.txt'
    fr_result = open(filename,'w')
    filename = './example_no_result.txt'
    fr_no_result = open(filename, 'w')
    for i in range(0,81262):
        if i in result_dict.keys():
            for one in result_dict[i]:
                print(one,end=' ',file=fr_result)
            print('',file=fr_result)
        else:
            if i in example_dict.keys():
                label = example_dict[i]['label']
                print(label, file=fr_result)
            else:
                print('167593', file=fr_result)
                # print('', file=fr_result)
                print(i,file=fr_no_result)
    fr_result.close()
    fr_no_result.close()


def generate_eval_result_file2():
    ''' train:456886,  max label:167593(11400)
        test:81262
    '''
    result_dict = {}
    filepath = './hier_result_leaf'
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.getsize(child):
            example_label_array = np.loadtxt(child, dtype=int)
            for i in range(0,np.size(example_label_array,axis=0)):
                example_index = example_label_array[i,0]
                label_index = example_label_array[i,1]
                if example_index in result_dict.keys():
                    if label_index not in result_dict[example_index]:
                        result_dict[example_index].append(label_index)
                else:
                    result_dict[example_index] = [label_index]

    example_dict = {}
    filepath = './hier_result_root'
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.getsize(child):
            fr = open(child,'r')
            lines = fr.readlines()
            fr.close()
            for line in lines:
                line = line.strip()
                linelist = line.split(' ')
                example = int(linelist[0])
                label = int(linelist[1])
                label_value = float(linelist[2])
                if example in example_dict.keys():
                    if label_value > example_dict[example]['label_value']:
                        example_dict[example]['label'] = label
                        example_dict[example]['label_value'] = label_value
                else:
                    example_dict[example] = {'label':label,'label_value':label_value}



    filename = './result.txt'
    fr_result = open(filename,'w')
    filename = './example_no_result.txt'
    fr_no_result = open(filename, 'w')
    for i in range(0,81262):
        if i in result_dict.keys():
            for one in result_dict[i]:
                print(one,end=' ',file=fr_result)
            print('',file=fr_result)
        else:
            if i in example_dict.keys():
                label = example_dict[i]['label']
                print(label, file=fr_result)
            else:
                print('167593', file=fr_result)
                # print('', file=fr_result)
                print(i,file=fr_no_result)
    fr_result.close()
    fr_no_result.close()




def main(argv=None):
    generate_hier_n_data_leaf2root()

if __name__ == '__main__':
    main()
