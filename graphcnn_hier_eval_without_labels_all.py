

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

            detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution_all')
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_predicted_value, fmt='%.4f')


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

    test_index_array = np.array(range(0, 81262))

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





