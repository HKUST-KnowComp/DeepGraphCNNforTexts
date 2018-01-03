
import numpy as np
import os
import graphcnn_option


filename = './hier_rootstr'
fr_rootstr = open(filename, 'w')
filename = './hier_rootlist'
fr_rootlist = open(filename, 'w')


THRESHOLD1 = 1200
THRESHOLD2 = 900
THRESHOLD3 = 900
THRESHOLD4 = 600
# SUM = 0
ROOT = 2143406  # the root of subgraph

DATA_PATH = graphcnn_option.DATA_PATH   # Path to data directory
TRAIN_DATA_DIR = graphcnn_option.TRAIN_DATA_DIR

def generate_labels_list_per_example(data_dir = DATA_PATH):
    """ get example(graph)-labels file:
    1 3 4
    6 9
    ...

    """
    filename = os.path.join(data_dir, 'data.train')
    fr = open(filename,'r')
    graphlines = fr.readlines()
    fr.close()
    filename = os.path.join(data_dir, 'example-labels')
    fr = open(filename, 'w')
    index = 1
    for line in graphlines:
        if index % 4 == 0:
            line = line.strip()  # remove the '\n',' ' on the head and end
            print(line, file=fr)
        index = index + 1
    fr.close()

def group_labels_by_examples(data_dir = DATA_PATH):
    '''According to the examples(graphs), labels can be divided into different groups.
       examples for each group are disjoint from other groups.

    Return:
        label groups list and example groups list
    '''

    label_groups_list = []
    example_groups_list = []

    # example-labels file
    filename = '../data/example-labels'
    fr = open(filename, 'r')
    example_labels_lines = fr.readlines()
    fr.close()
    examples_number = len(example_labels_lines)
    labels_number = 36504
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

    filename = '../data/label_groups'
    fr = open(filename, 'w')
    for list_i in label_groups_list:
        for i in list_i:
            print(i,end=' ',file=fr)
        print('',file=fr)
    fr.close()

    filename = '../data/example_groups'
    fr = open(filename, 'w')
    for list_i in example_groups_list:
        for i in list_i:
            print(i, end=' ', file=fr)
        print('', file=fr)
    fr.close()

# 生成原始的样本标签
def generate_example_labels_orig():

    filename = '../data/remap'
    remap = np.loadtxt(filename, dtype=int)
    filename = '../data/example_labels_remap'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    filename = '../data/example_labels_orig'
    fr = open(filename, 'w')
    for line in lines:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [int(j) for j in linelist]
        labels_remap = remap[linelist, 0]
        for i in labels_remap:
            print(i, end=' ', file=fr)
        print('', file=fr)
    fr.close()

# 将hierarchy的父子关系合并，统计每个Node的所有直接父亲和孩子
def generate_hier_dict_with_parent_child():

    filename = '../data/lshtc/hierarchyWikipediaMedium.txt'
    parent_child_array = np.loadtxt(filename,dtype=int)

    hier_dict = {}
    for i in range(0,np.size(parent_child_array,0)):
        parent = parent_child_array[i][0]
        child = parent_child_array[i][1]
        if parent in hier_dict.keys():
            hier_dict[parent][1].append(child)
        else:
            parent_list = []
            child_list = [child]
            hier_dict[parent] = [parent_list, child_list]
        if child in hier_dict.keys():
            hier_dict[child][0].append(parent)
        else:
            parent_list = [parent]
            child_list = []
            hier_dict[child] = [parent_list, child_list]
    filename = '../data/hier_relation'
    fr = open(filename,'w')
    for k,v in hier_dict.items():
        print(k,file=fr)
        for hier_list in v:
            for label in hier_list:
                print(label,end=' ',file=fr)
            print('',file=fr)
    fr.close()
    sum=0
    for k, v in hier_dict.items():
        if len(v[0]) != 0:
            for i in range(0, np.size(parent_child_array, 0)):
                if parent_child_array[i][0] in v[0] and parent_child_array[i][1] in v[0]:
                    if len(v[1])!=0:
                        print(True)
                        return

    print(sum)
    return hier_dict

# 生成扩展后的样本标签
def generate_example_labels_orig_expand():
    filename = '../data/example_labels_orig'
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    hier_dict = generate_hier_dict_with_parent_child()
    filename = '../data/example_labels_orig_expand'
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
# root: 2143406
def generate_hier0_data():  # 2143406
    ''' hier0_remap,hier0_labels,hier0_graphs_index,hier0_graphs'''
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

    hier0_remap = []
    # find root nodes:(who has no parent)
    for k in hier_dict.keys():
        if len(hier_dict[k][0])==0:
            hier0_remap.append(k)

    hier0_remap_len = len(hier0_remap)
    filename = '../data/lshtc/hier0_remap'
    fr = open(filename,'w')
    for i in range(0,hier0_remap_len):
        print('%d %d' % (hier0_remap[i],i), file=fr)
    fr.close()

# 生成root为根的子树的叶子节点列表
def _compute_leaf_node_of_sub_tree(hier_dict,root,fr):
    hier_remap = hier_dict[root][1]
    hier_remap_len = len(hier_remap)
    if hier_remap_len == 0:  # leaf node
        print('%d %d'%(root,root),file=fr)
        return [root]
    leaf_root = []
    for label in hier_remap:
        leaf_list = _compute_leaf_node_of_sub_tree(hier_dict,label,fr)
        for one in leaf_list:
            if one not in leaf_root:
                leaf_root.append(one)
    print('%d' % (root), end=' ', file=fr)
    for one in leaf_root:
        print('%d' % (one), end=' ', file=fr)
    print('',file=fr)
    return leaf_root


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



def _subfunc1_compute_leaf_node_of_sub_tree(hier_dict,root,hier_leafList_dict):
    '''计算以root为根节点的树的叶节点列表'''
    if root in hier_leafList_dict.keys():
        return
    hier_remap = hier_dict[root][1]
    hier_remap_len = len(hier_remap)
    if hier_remap_len == 0:  # leaf node
        hier_leafList_dict[root]=[root]
        return
    leaf_root = []
    for label in hier_remap:
        _subfunc1_compute_leaf_node_of_sub_tree(hier_dict,label,hier_leafList_dict)
        leaf_list = hier_leafList_dict[label]
        for one in leaf_list:
            if one not in leaf_root:
                leaf_root.append(one)
    hier_leafList_dict[root] = leaf_root
    return

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
    if len(root_list) > 3:
        filename = '../data/lshtc/hier_%d_%d_others_'%(root_list[0],root_list[1])+'rootstr'
        fr = open(filename, 'w')
        print(root_str,file=fr)
        fr.close()
        root_str = '_%d_%d_others_'%(root_list[0],root_list[1])
        # filename = '../data/lshtc/hier_%d_%d_others_'%(root_list[0],root_list[1])+'remap'
    filename = '../data/lshtc/hier'+root_str+'remap'
    fr = open(filename, 'w')
    for i in range(0, hier_remap_len):
        print('%d %d' % (hier_remap[i], i), file=fr)
    fr.close()
    filename = '../data/lshtc/hier'+root_str+'labels'
    fr_label = open(filename, 'w')
    filename = '../data/lshtc/hier'+root_str+'graphs_index'
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


# def _subfunc2_generate_hier_remap(hier_leafList_dict, root_list):
#     hier_remap = []
#     root_str = '_'
#     for root in root_list:
#         print(root,end=' ',file=fr_rootlist)
#         root_str += '%d_' % root
#         root_leaf = hier_leafList_dict[root]
#         for one in root_leaf:
#             if one not in hier_remap:
#                 hier_remap.append(one)
#     hier_remap_len = len(hier_remap)
#     if len(root_list) > 3:
#         root_str = '_%d_%d_others_'%(root_list[0],root_list[1])
#     print(root_str,file = fr_rootstr)
#     print('',file = fr_rootlist)
#


def _subfunc4(hier_dict,root,hier_leafList_dict,level):
    if len(hier_dict[root][1])==0:
        return
    leafList = hier_leafList_dict[root]
    print('%d %d'%(root,len(leafList)))
    if len(leafList) <=500:
        _subfunc5(root,leafList,level)
        return
    _subfunc5(root, hier_dict[root][1], level)

    for one in hier_dict[root][1]:
        _subfunc4(hier_dict,one,hier_leafList_dict,level+1)

def _subfunc5(root, leafList, level):
    hier_remap = leafList
    root_str = '_%d_%d_'%(level,root)

    hier_remap_len = len(hier_remap)

    filename = '../data/lshtc/hier'+root_str+'remap'
    fr = open(filename, 'w')
    for i in range(0, hier_remap_len):
        print('%d %d' % (hier_remap[i], i), file=fr)
    fr.close()
    filename = '../data/lshtc/hier'+root_str+'labels'
    fr_label = open(filename, 'w')
    filename = '../data/lshtc/hier'+root_str+'graphs_index'
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
            hier_leafList_dict.clear()
            _subfunc1_compute_leaf_node_of_sub_tree(hier_dict, ROOT, hier_leafList_dict)

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
                hier_leafList_dict.clear()
                _subfunc1_compute_leaf_node_of_sub_tree(hier_dict, ROOT, hier_leafList_dict)
        # if flag == 1:
        #     tmp_root_list = []
        #     for one in child_list:
        #         father = hier_dict[one][0]
        #         for two in father:
        #             if two not in tmp_root_list:
        #                 tmp_root_list.append(two)
        #     for tmp_root in tmp_root_list:
        #         tmp_child_list = hier_dict[tmp_root][1]
        #         tmp_root_leaf = []
        #         for one in tmp_child_list:
        #             for two in hier_leafList_dict[one]:
        #                 if two not in tmp_root_leaf:
        #                     tmp_root_leaf.append(two)
        #         hier_leafList_dict[tmp_root] = tmp_root_leaf
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
            hier_leafList_dict.clear()
            _subfunc1_compute_leaf_node_of_sub_tree(hier_dict, ROOT, hier_leafList_dict)
    # print(len(hier_leafList_dict[root]))

def generate_hier_n_data_leaf2root():
    '''
        hier_XXX_remap
        hier_XXX_labels
        hier_XXX_graphs_index
        hier_XXX_graphs
    '''

    filename = '../data/lshtc/hier_relation'
    fr = open(filename, 'r')
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
    root = ROOT
    hier_leafList_dict = {}
    # _subfunc1_compute_leaf_node_of_sub_tree(hier_dict,root,hier_leafList_dict)

    filename = '../data/lshtc/hier_2143406_remap'
    root_remap = np.loadtxt(filename,dtype=int)
    root_labels = root_remap[:,0]
    for one in root_labels:
        one_child_list = hier_dict[one][1]
        hier_dict[one][1] = []
        for one_child in one_child_list:
            if one in hier_dict[one_child][0]:
                hier_dict[one_child][0].remove(one)
    # hier_leafList_dict.clear()
    _subfunc1_compute_leaf_node_of_sub_tree(hier_dict, ROOT, hier_leafList_dict)

    _subfunc4(hier_dict,ROOT,hier_leafList_dict,0)











def main(argv=None):
    generate_hier_n_data_leaf2root()
if __name__ == '__main__':
    main()
