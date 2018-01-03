

# 444

from datetime import datetime
import math
import time
import os
import shutil

import numpy as np
import tensorflow as tf

import graphcnn_model
import graphcnn_input
import graphcnn_option

EVALUTION_THRESHOLD_FOR_MULTI_LABEL = 0.9

evalDataSet = None

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tmp/graphcnn_hier_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/graphcnn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")




def generate_eval_index():
    test_index_array = []
    # filepath = os.path.join(graphcnn_option.DATA_PATH, graphcnn_option.HIER_DIR_NAME)
    filepath = '../hier_eval_root'
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.getsize(child):
            example_label_array = np.loadtxt(child,dtype=int)
            examlpe_array = example_label_array[:,0]
            label_array = example_label_array[:, 1]
            for root in graphcnn_option.HIER_ROOT_CODE:
                index = np.where(label_array==root)[0]
                for one in examlpe_array[index]:
                    if one not in test_index_array:
                        test_index_array.append(one)

    # for allDir in pathDir:
    #     child = os.path.join(filepath, allDir)
    #     os.remove(child)


    filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_hier_eval_index')
    np.savetxt(filename,test_index_array,fmt='%d')

    return test_index_array


def evaluate(checkpoint,test_index_array):
        detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution_all')
        total_predicted_value = np.loadtxt(detail_filename,dtype=float)
        total_predicted_value = total_predicted_value[test_index_array]

        total_predicted_value_max = np.max(total_predicted_value, axis=1)
        total_predicted_value_argmax = np.argmax(total_predicted_value, axis=1)
        total_predicted_value = (
        (total_predicted_value) >= EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)

        detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
        if os.path.exists(detail_filename):
            os.remove(detail_filename)
        np.savetxt(detail_filename, total_predicted_value, fmt='%d')


        filename = os.path.join(graphcnn_option.EVAL_DATA_DIR, graphcnn_option.DATA_LABELS_REMAP_NAME)
        total_remap = np.loadtxt(filename, dtype=int)

        detail_filename = os.path.join(graphcnn_option.EVAL_DATA_DIR, graphcnn_option.HIER_DIR_NAME,
                                       graphcnn_option.HIER_labels_remap_file)
        remap = np.loadtxt(detail_filename, dtype=int)

        filename = os.path.join('../hier_result_leaf', graphcnn_option.HIER_eval_result_leaf_file)
        fr_leaf = open(filename,'a')
        filename = os.path.join('../hier_result_leaf_exp', graphcnn_option.HIER_eval_result_leaf_exp_file)
        fr_leaf_exp = open(filename, 'a')
        filename = os.path.join('../hier_result_root', graphcnn_option.HIER_eval_result_root_file)
        fr_root = open(filename, 'w')

        # rootstr_tmp = []
        detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_list')
        fr = open(detail_filename, 'w')
        for i in range(0, np.size(total_predicted_value, axis=0)):
            labels = np.where(total_predicted_value[i] == 1)[0]
            if len(labels) > 0:
                labels_remap = remap[labels, 0]
                for elem in labels_remap:
                    print(elem, end=' ', file=fr)
                    if elem in total_remap[:,0]: # leaf
                        print('%d %d'%(test_index_array[i],elem),file=fr_leaf)
                    else:
                        print('%d %d' % (test_index_array[i], elem), file=fr_root)
                        # for j in range(0,len(rootlist)):
                        #     if elem in rootlist[j]:
                        #         if rootstr[j] not in rootstr_tmp:
                        #             rootstr_tmp.append(rootstr[j])
                print('', file=fr)
            else:
                # labels_remap = remap[:, 0]
                labels = total_predicted_value_argmax[i]
                labels_value = total_predicted_value_max[i]
                labels_remap = remap[labels, 0]
                # for elem in labels_remap:
                elem = labels_remap
                print(elem, file=fr)
                if elem in total_remap[:, 0]:  # leaf
                    print('%d %d %.4f' % (test_index_array[i], elem, labels_value), file=fr_leaf_exp)
                else:
                    print('%d %d' % (test_index_array[i], elem), file=fr_root)
                # if labels_value < 0.5:
                #     labels_remap = remap[:, 0]
                #     for elem in labels_remap:
                #         if elem not in total_remap[:, 0]:
                #             print('%d %d' % (test_index_array[i], elem), file=fr_root)

        fr.close()
        fr_leaf.close()
        fr_root.close()
        fr_leaf_exp.close()

        # filename = os.path.join(FLAGS.eval_dir, 'hier_next_root')
        # fr = open(filename, 'w')
        # for one in rootstr_tmp:
        #     print(one)
        #     print(one,file=fr)
        # fr.close()




def main(argv=None):  # pylint: disable=unused-argument
    global evalDataSet
    # assert not tf.gfile.Exists(FLAGS.eval_dir), 'please move the old evaluate directory to pre_versions!'

    # test_index_array = np.array(range(0, 81262))
    if graphcnn_option.HIER_ROOT_CODE[0]==2143406: # root
        test_index_array = np.array(range(0,81262))
        # test_index_array = np.loadtxt('../example_no_result.txt',dtype=int)
    else:
        test_index_array = generate_eval_index()
    if test_index_array is None or len(test_index_array)==0:
        print('no hier_data need eval')
        return
    else:
        print('choosing for evaluation...')
        print('choosed number:%d' % len(test_index_array))

    # checkpoint = input('please input the choosed checkpoint to eval:(0 for latest)')
    checkpoint = '0'

    # print('choosing for evaluation...')
    evaluate(checkpoint,test_index_array)


if __name__ == '__main__':
    tf.app.run()





