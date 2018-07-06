
# HR-SVM

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import math

import graphcnn_input
import graphcnn_option
import SVM_model



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tmp/graphcnn_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_epochs', 8000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


# max_steps for train:
STEPS_PER_ECOPH = None
MAX_STEPS = None
# the period to save the model checkpoint.
CKPT_PERIOD = None

trainDataSet = None

   
def train(newTrain,checkpoint):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        data = tf.placeholder(tf.float32, [graphcnn_input.TRAIN_BATCH_SIZE, graphcnn_input.NUM_CHANNELS])  # NUM_CHANNELS: feature dim
        labels = tf.placeholder(tf.int32, [graphcnn_input.TRAIN_BATCH_SIZE,graphcnn_input.NUM_CLASSES])  # with value: -1,1

        # inference model.
        logits = SVM_model.SVM_inference(data, labels)
        
        # Declare prediction function
        prediction = tf.sign(logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

        # Calculate loss.
        loss = SVM_model.SVM_loss()

        # updates the model parameters.
        train_op = SVM_model.SVM_train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(var_list=tf.global_variables(),
                               max_to_keep=6,
                               keep_checkpoint_every_n_hours=10)

        if graphcnn_option.SUMMARYWRITER:
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        first_step = 0
        if not newTrain:
            if checkpoint == '0': # choose the latest one
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
                    # Restores from checkpoint
                    new_saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step_for_restore = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    first_step = int(global_step_for_restore) + 1
                else:
                    print('No checkpoint file found')
                    return
            else: #
                if os.path.exists(os.path.join(FLAGS.train_dir, 'model.ckpt-' + checkpoint)):
                    new_saver = tf.train.import_meta_graph(
                        os.path.join(FLAGS.train_dir, 'model.ckpt-' + checkpoint + '.meta'))
                    new_saver.restore(sess,
                        os.path.join(FLAGS.train_dir, 'model.ckpt-' + checkpoint))
                    first_step = int(checkpoint) + 1
                else:
                    print('No checkpoint file found')
                    return
        else:
            sess.run(init)

        if graphcnn_option.SUMMARYWRITER:
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        filename_train_log = os.path.join(FLAGS.train_dir, 'log_train')
        if os.path.exists(filename_train_log):
            file_train_log = open(filename_train_log, 'a')
        else:
            file_train_log = open(filename_train_log, 'w')

        # learning_rate = graphcnn_option.lr_decay_value[0]  # 0.1(5), 0.01(100), 0.001(500), 0.0001(300), 0.00001(100)
        # learning_rate_index = 0
        for step in range(first_step,MAX_STEPS):
            # if learning_rate_index < len(graphcnn_option.lr_decay_value) - 1:
            #     if step > STEPS_PER_ECOPH * graphcnn_option.lr_decay_ecophs[learning_rate_index]:
            #         learning_rate_index = learning_rate_index + 1
            #         learning_rate = graphcnn_option.lr_decay_value[learning_rate_index]

            train_data, train_label = trainDataSet.next_batch(graphcnn_input.TRAIN_BATCH_SIZE)
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict= {data:train_data, labels:train_label})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                sec_per_batch = float(duration)
                format_str = ('%s: step=%d, loss=%.4f; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, sec_per_batch), file=file_train_log)
                print(format_str % (datetime.now(), step, loss_value, sec_per_batch))

            if graphcnn_option.SUMMARYWRITER:
                if step % 100 == 0:
                    summary_str = sess.run(summary_op,
                                           feed_dict= {data:train_data, labels:train_label})
                    summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically. (named 'model.ckpt-global_step.meta')
            if step % CKPT_PERIOD == 0 or (step + 1) == MAX_STEPS:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        file_train_log.close()

def main(argv=None):
    global trainDataSet, evalDataSet, STEPS_PER_ECOPH, MAX_STEPS, CKPT_PERIOD
    newTrain = True
    checkpoint = 0
    # assert not tf.gfile.Exists(FLAGS.train_dir), 'please move the old train directory to pre_versions!'
    if tf.gfile.Exists(FLAGS.train_dir):
        ans = input('whether to open up a new training:(y/n)')
        if ans == 'y' or ans == 'Y':
            newTrain = True
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        elif ans == 'n' or ans == 'N':
            newTrain = False
            checkpoint = input('please input the choosed checkpoint to restore:(0 for latest)')
        else:
            print('invalid input!')
            return
    if newTrain:
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # update paras
    trainDataSet = graphcnn_input.generate_SVM_train_data(graphcnn_option.TRAIN_DATA_DIR,
                                                      ont_hot=True,index_mode=True)

    # max_steps for train:
    STEPS_PER_ECOPH = math.ceil(
        graphcnn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / float(graphcnn_input.TRAIN_BATCH_SIZE))
    MAX_STEPS = FLAGS.max_epochs * STEPS_PER_ECOPH

    # the period to save the model checkpoint.
    CKPT_PERIOD = graphcnn_option.CKPT_PERIOD  # ?????????????????????
    # CKPT_PERIOD = 5000
    # tem = str(STEPS_PER_ECOPH * 20)  # save the model every ecoph  # 5
    # CKPT_PERIOD = int(int(tem[0]) * pow(10, len(tem) - 1))

    print('training...')
    train(newTrain,checkpoint)


if __name__ == '__main__':
    tf.app.run()







