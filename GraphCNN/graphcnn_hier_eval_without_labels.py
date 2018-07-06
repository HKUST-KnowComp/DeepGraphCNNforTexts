

# 222

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



EVALUTION_THRESHOLD_FOR_MULTI_LABEL = 0.5

# 生成测试数据的索引文件
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

    # 将索引文件写到hier_eval文件夹下
    filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_hier_eval_index')
    np.savetxt(filename,test_index_array,fmt='%d')

    return test_index_array


def evaluate(checkpoint,test_index_array):
    with tf.Graph().as_default() as g, tf.device('/cpu:0'):
        # Get images and labels
        data = tf.placeholder(tf.float32, [graphcnn_input.EVAL_BATCH_SIZE, graphcnn_input.HEIGHT, graphcnn_input.WIDTH,
                                           graphcnn_input.NUM_CHANNELS])
        # labels = tf.placeholder(tf.int32, [graphcnn_input.EVAL_BATCH_SIZE,graphcnn_input.NUM_CLASSES])

        # inference
        logits = graphcnn_model.inference(data, eval_data=True)
        # logits = graphcnn_model.inference_CPU(data, eval_data=True, dependencies_loss=False)

        # multi-label sigmoid
        logits = tf.sigmoid(logits)

        # Restore the moving average version of the learned variables for eval. # ?????????????????????????
        variable_averages = tf.train.ExponentialMovingAverage(graphcnn_option.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        # summary_op = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)


        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement)) as sess:
            if checkpoint == '0':
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # extract global_step
                    global_step_for_restore = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                else:
                    print('No checkpoint file found')
                    return
            else:
                if os.path.exists(os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-' + checkpoint)):
                    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-' + checkpoint))
                    global_step_for_restore = int(checkpoint)
                else:
                    print('No checkpoint file found')
                    return

            num_iter = int(math.floor(graphcnn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / graphcnn_input.EVAL_BATCH_SIZE))
            total_sample_count = num_iter * graphcnn_input.EVAL_BATCH_SIZE
            step = 0
            total_predicted_value = np.zeros([1, graphcnn_input.NUM_CLASSES], dtype=np.float32)  ##
            while step < num_iter:
                test_data = evalDataSet.next_batch(graphcnn_input.EVAL_BATCH_SIZE)
                predicted_value = sess.run(
                    logits, feed_dict={data: test_data})
                total_predicted_value = np.concatenate((total_predicted_value, predicted_value), axis=0)
                step += 1

            total_predicted_value = total_predicted_value[1:]

            detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution')
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_predicted_value, fmt='%.4f')
            total_predicted_value_argmax = np.argmax(total_predicted_value, axis=1)
            total_predicted_value = (
            (total_predicted_value) >= EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)
            assert total_sample_count == total_predicted_value.shape[0], 'sample_count error!'
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
            filename = os.path.join('../hier_result_root', graphcnn_option.HIER_eval_result_root_file)
            fr_root = open(filename, 'w')

            # filename = os.path.join(graphcnn_option.EVAL_DATA_DIR, 'hier_rootstr')
            # fr = open(filename, 'r')
            # rootstr = fr.readlines()
            # fr.close()
            # filename = os.path.join(graphcnn_option.EVAL_DATA_DIR, 'hier_rootlist')
            # fr = open(filename, 'r')
            # rootlines = fr.readlines()
            # fr.close()
            # rootlist = []
            # for line in rootlines:
            #     line = line.strip()
            #     linelist = line.split(' ')
            #     linelist = [int(k) for k in linelist]
            #     rootlist.append(linelist)

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
                    labels_remap = remap[labels, 0]
                    for elem in labels_remap:
                        print(elem, end=' ', file=fr)
                        if elem in total_remap[:, 0]:  # leaf
                            print('%d %d' % (test_index_array[i], elem), file=fr_leaf)
                        else:
                            print('%d %d' % (test_index_array[i], elem), file=fr_root)
                            # for j in range(0,len(rootlist)):
                            #     if elem in rootlist[j]:
                            #         if rootstr[j] not in rootstr_tmp:
                            #             rootstr_tmp.append(rootstr[j])
                    print('', file=fr)
            fr.close()
            fr_leaf.close()
            fr_root.close()

            # filename = os.path.join(FLAGS.eval_dir, 'hier_next_root')
            # fr = open(filename, 'w')
            # for one in rootstr_tmp:
            #     print(one)
            #     print(one,file=fr)
            # fr.close()

            filename_eval_log = os.path.join(FLAGS.eval_dir, 'log_eval')
            file_eval_log = open(filename_eval_log, 'w')
            np.set_printoptions(threshold=np.nan)
            print('\nevaluation:', file=file_eval_log)
            print('\nevaluation:')
            print('  %s, ckpt-%d' % (datetime.now(), global_step_for_restore), file=file_eval_log)
            print('  %s, ckpt-%d' % (datetime.now(), global_step_for_restore))
            print('evaluation is end...')
            print('evaluation is end...', file=file_eval_log)

            print('evaluation samples number:%d, evaluation classes number:%d' %
                  (total_predicted_value.shape[0], total_predicted_value.shape[1]), file=file_eval_log)
            print('evaluation samples number:%d, evaluation classes number:%d' %
                  (total_predicted_value.shape[0], total_predicted_value.shape[1]))
            print('evaluation detail: '
                  + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
                  + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution'),
                  file=file_eval_log)
            print('evaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval')
                  + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
                  + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution'))
            file_eval_log.close()


def main(argv=None):  # pylint: disable=unused-argument
    global evalDataSet
    # assert not tf.gfile.Exists(FLAGS.eval_dir), 'please move the old evaluate directory to pre_versions!'

    if tf.gfile.Exists(FLAGS.eval_dir):
        # print('the evaluate data has already exists!')
        # str = input('continue will delete the old evaluate directory:(y/n)')
        # if str == 'y' or str == 'Y':
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
        #elif str == 'n' or str == 'N':
        #    print('eval end!')
        #    return
        #else:
        #    print('invalid input!')
        #    return
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # test_index_array = np.array(range(0, 81262))
    if graphcnn_option.HIER_ROOT_CODE[0]==2143406: # root
        test_index_array = np.array(range(0,81262))
        # test_index_array = np.loadtxt('../example_no_result.txt',dtype=int)
    else:
        test_index_array = generate_eval_index()
    if test_index_array is None or len(test_index_array)==0:
        print('no hier_data need eval')
        return

    # checkpoint = input('please input the choosed checkpoint to eval:(0 for latest)')
    checkpoint = '0'
    evalDataSet = graphcnn_input.generate_hier_eval_data(test_index_array,
                                                         data_dir=graphcnn_option.EVAL_DATA_DIR,
                                                         ont_hot=True,
                                                         index_mode=True,
                                                         label_used=False)
    print('evaluating...')
    evaluate(checkpoint,test_index_array)


if __name__ == '__main__':
    tf.app.run()





