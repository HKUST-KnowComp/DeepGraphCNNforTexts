

from datetime import datetime
import math
import time
import os
import shutil

import numpy as np
import tensorflow as tf

import graphcnn
import graphcnn_input

EVAL_DATA_DIR = graphcnn_input.TRAIN_DATA_DIR
evalDataSet = None

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tmp/graphcnn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/graphcnn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60*5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")



def evaluate(checkpoint):
    with tf.Graph().as_default() as g:
        # Get images and labels
        data = tf.placeholder(tf.float32, [graphcnn_input.EVAL_BATCH_SIZE, graphcnn_input.HEIGHT, graphcnn_input.WIDTH,
                                                 graphcnn_input.NUM_CHANNELS])
        labels = tf.placeholder(tf.int32, [graphcnn_input.EVAL_BATCH_SIZE])

        # inference
        logits = graphcnn.inference(data, eval_data=True)

        # Calculate predictions of top k
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

        # Restore the moving average version of the learned variables for eval. # ?????????????????????????
        variable_averages = tf.train.ExponentialMovingAverage(graphcnn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        # summary_op = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        best_eval_value = 0;   ####
        best_eval_ckpt = 0;  ####

        with tf.Session() as sess:
            # tf.train.update_checkpoint_state(FLAGS.checkpoint_dir,os.path.join(FLAGS.checkpoint_dir,'model.ckpt-10000'))
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
            true_count_top1 = 0  # Counts the number of correct predictions.
            true_count_top5 = 0
            total_sample_count = num_iter * graphcnn_input.EVAL_BATCH_SIZE
            step = 0
            total_predicted_value = np.array([[-1, -1, -1, -1, -1]])
            total_true_value = np.array([])
            while step < num_iter:
                test_data, test_label = evalDataSet.next_batch(graphcnn_input.EVAL_BATCH_SIZE)
                predictions_top1, predictions_top5, predicted_value, true_value = sess.run(
                    [top_1_op, top_5_op, logits, labels], feed_dict={data: test_data, labels: test_label})
                true_count_top1 += np.sum(predictions_top1)
                true_count_top5 += np.sum(predictions_top5)
                argmax = np.argmax(predicted_value, axis=1)
                argmax_top5 = np.reshape(argmax, [-1, 1])
                for top_i in range(1, 5):
                    for j in range(np.size(argmax)):
                        predicted_value[j, argmax[j]] = -1
                    argmax = np.argmax(predicted_value, axis=1)
                    argmax_top5 = np.concatenate((argmax_top5, np.reshape(argmax, [-1, 1])), axis=1)
                total_predicted_value = np.concatenate((total_predicted_value, argmax_top5), axis=0)
                total_true_value = np.append(total_true_value, true_value)
                step += 1
            total_predicted_value = total_predicted_value[1:]
            total_true_value = np.reshape(total_true_value, [-1, 1])

            np.set_printoptions(threshold=np.nan)
            filename_eval_log = os.path.join(FLAGS.eval_dir, 'log_eval')
            file_eval_log = open(filename_eval_log, 'w')


            accuracy = float(true_count_top1) / total_sample_count
            accuracy_top1 = accuracy
            print('\nevaluation:(top-1)', file=file_eval_log)
            print('\nevaluation:(top-1)')
            print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore),
                  file=file_eval_log)
            print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore))
            print('true_count / total_sample_count  :  %d / %d' % (true_count_top1, total_sample_count),
                  file=file_eval_log)
            print('true_count / total_sample_count  :  %d / %d' % (true_count_top1, total_sample_count))
            print('Class Statistics:', file=file_eval_log)
            for label_i in range(0, graphcnn_input.NUM_CLASSES):
                prediction = total_predicted_value[:, 0] == label_i
                expectation = total_true_value[:, 0] == label_i
                equal = prediction & expectation
                preNum = sum(prediction)
                expecNum = sum(expectation)
                equal_num = sum(equal)
                precise = equal_num / float(preNum)
                recall = equal_num / float(expecNum)
                F1 = ((precise * recall) * 2) / (precise + recall)
                print('class %d: precise=%.4f (%d/%d), recall=%.4f (%d/%d), F1-Measure=%.4f' %
                      (label_i, precise, equal_num, preNum, recall, equal_num, expecNum, F1),
                      file=file_eval_log)

            accuracy = float(true_count_top5) / total_sample_count
            print('\nevaluation:(top-5)', file=file_eval_log)
            print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore),
                  file=file_eval_log)
            print('true_count / total_sample_count  :  %d / %d' % (true_count_top5, total_sample_count),
                  file=file_eval_log)
            print('\nevaluation:(top-5)')
            print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore))
            print('true_count / total_sample_count  :  %d / %d' % (true_count_top5, total_sample_count))
            # print('Class Statistics:', file=file_eval_log)
            # for label_i in range(0, graphcnn_input.NUM_CLASSES):
            #     prediction = np.sum(total_predicted_value == label_i, axis=1)
            #     prediction.astype(bool)
            #     expectation = total_true_value[:, 0] == label_i
            #     equal = prediction & expectation
            #     preNum = sum(prediction)
            #     expecNum = sum(expectation)
            #     equal_num = sum(equal)
            #     precise = equal_num / float(preNum)
            #     recall = equal_num / float(expecNum)
            #     F1 = ((precise * recall) * 2) / (precise + recall)
            #     print('class %d: precise=%.4f (%d/%d), recall=%.4f (%d/%d), F1-Measure=%.4f' %
            #           (label_i, precise, equal_num, preNum, recall, equal_num, expecNum, F1),
            #           file=file_eval_log)

            # print('\nevaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_detail'),
            #       file=file_eval_log)
            # print('\nevaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval') + ', ' +
            #       os.path.join(FLAGS.eval_dir, 'log_eval_for_detail') + '\n')

            file_eval_log.close()

            detail = np.concatenate((total_predicted_value, total_true_value), axis=1)
            detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_detail')
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, detail, fmt='%d')

            best_eval_ckpt = global_step_for_restore
            best_eval_value = accuracy_top1
            sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval')
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)
            sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_detail')
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_detail')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)
            sourceFile = ckpt.model_checkpoint_path
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval.ckpt')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)

            # summary = tf.Summary()
            # summary.ParseFromString(sess.run(summary_op))
            # summary.value.add(tag='Precision @ 1', simple_value=precision)
            # summary_writer.add_summary(summary, global_step_for_restore)

        while True:
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # extract global_step
                    global_step_for_restore = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    if global_step_for_restore > best_eval_ckpt:
                        # Restores from checkpoint
                        saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                    return

                if global_step_for_restore > best_eval_ckpt:
                    num_iter = int(math.floor(graphcnn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / graphcnn_input.EVAL_BATCH_SIZE))
                    true_count_top1 = 0  # Counts the number of correct predictions.
                    true_count_top5 = 0
                    total_sample_count = num_iter * graphcnn_input.EVAL_BATCH_SIZE
                    step = 0
                    total_predicted_value = np.array([[-1,-1,-1,-1,-1]])
                    total_true_value = np.array([])
                    while step < num_iter:
                        test_data, test_label = evalDataSet.next_batch(graphcnn_input.EVAL_BATCH_SIZE)
                        predictions_top1, predictions_top5, predicted_value, true_value = sess.run(
                            [top_1_op, top_5_op, logits, labels], feed_dict={data: test_data, labels: test_label})
                        true_count_top1 += np.sum(predictions_top1)
                        true_count_top5 += np.sum(predictions_top5)
                        argmax = np.argmax(predicted_value, axis=1)
                        argmax_top5 = np.reshape(argmax,[-1,1])
                        for top_i in range(1,5):
                            for j in range(np.size(argmax)):
                                predicted_value[j,argmax[j]] = -1
                            argmax = np.argmax(predicted_value, axis=1)
                            argmax_top5 = np.concatenate((argmax_top5,np.reshape(argmax,[-1,1])),axis=1)
                        total_predicted_value = np.concatenate((total_predicted_value,argmax_top5),axis=0)
                        total_true_value = np.append(total_true_value, true_value)
                        step += 1
                    total_predicted_value = total_predicted_value[1:]
                    total_true_value = np.reshape(total_true_value, [-1, 1])
                    filename_eval_log = os.path.join(FLAGS.eval_dir, 'log_eval')
                    file_eval_log = open(filename_eval_log, 'a')
                    np.set_printoptions(threshold=np.nan)

                    accuracy = float(true_count_top1) / total_sample_count
                    accuracy_top1 = accuracy
                    print('\nevaluation:(top-1)',file=file_eval_log)
                    print('\nevaluation:(top-1)')
                    print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore),
                          file=file_eval_log)
                    print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore))
                    print('true_count / total_sample_count  :  %d / %d' % (true_count_top1, total_sample_count),
                          file=file_eval_log)
                    print('true_count / total_sample_count  :  %d / %d' % (true_count_top1, total_sample_count))
                    print('Class Statistics:',file=file_eval_log)
                    for label_i in range(0, graphcnn_input.NUM_CLASSES):
                        prediction = total_predicted_value[:,0] == label_i
                        expectation = total_true_value[:,0] == label_i
                        equal = prediction & expectation
                        preNum = sum(prediction)
                        expecNum = sum(expectation)
                        equal_num = sum(equal)
                        precise = equal_num / float(preNum)
                        recall = equal_num / float(expecNum)
                        F1 = ((precise*recall) * 2) / (precise + recall)
                        print('class %d: precise=%.4f (%d/%d), recall=%.4f (%d/%d), F1-Measure=%.4f' %
                              (label_i,precise,equal_num,preNum,recall,equal_num,expecNum,F1),
                              file=file_eval_log )

                    accuracy = float(true_count_top5) / total_sample_count
                    print('\nevaluation:(top-5)', file=file_eval_log)
                    print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore),
                          file=file_eval_log)
                    print('true_count / total_sample_count  :  %d / %d' % (true_count_top5, total_sample_count),
                          file=file_eval_log)
                    print('\nevaluation:(top-5)')
                    print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore))
                    print('true_count / total_sample_count  :  %d / %d' % (true_count_top5, total_sample_count))
                    # print('Class Statistics:', file=file_eval_log)
                    # for label_i in range(0, graphcnn_input.NUM_CLASSES):
                    #     prediction = np.sum(total_predicted_value == label_i, axis=1)
                    #     prediction.astype(bool)
                    #     expectation = total_true_value[:,0] == label_i
                    #     equal = prediction & expectation
                    #     preNum = sum(prediction)
                    #     expecNum = sum(expectation)
                    #     equal_num = sum(equal)
                    #     precise = equal_num / float(preNum)
                    #     recall = equal_num / float(expecNum)
                    #     F1 = ((precise*recall) * 2) / (precise + recall)
                    #     print('class %d: precise=%.4f (%d/%d), recall=%.4f (%d/%d), F1-Measure=%.4f' %
                    #           (label_i,precise,equal_num,preNum,recall,equal_num,expecNum,F1),
                    #           file=file_eval_log )

                    # print('\nevaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_detail'),
                    #       file=file_eval_log)
                    # print('\nevaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval') + ', ' +
                    #       os.path.join(FLAGS.eval_dir, 'log_eval_for_detail') + '\n')

                    file_eval_log.close()

                    detail = np.concatenate((total_predicted_value, total_true_value), axis=1)
                    detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_detail')
                    if os.path.exists(detail_filename):
                        os.remove(detail_filename)
                    np.savetxt(detail_filename,detail,fmt='%d')

                    if accuracy_top1 > best_eval_value:
                        best_eval_value = accuracy_top1

                        filename_eval_best = os.path.join(FLAGS.eval_dir, 'best_eval')
                        file_eval_best = open(filename_eval_best, 'w')
                        print('evaluation(top-1):', file=file_eval_best)
                        print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy_top1, global_step_for_restore),
                              file=file_eval_best)
                        print('true_count / total_sample_count  :  %d / %d' % (true_count_top1, total_sample_count),
                              file=file_eval_best)
                        print('evaluation(top-5):', file=file_eval_best)
                        print('%s: accuracy=%.4f; ckpt-%d' % (datetime.now(), accuracy, global_step_for_restore),
                              file=file_eval_best)
                        print('true_count / total_sample_count  :  %d / %d' % (true_count_top5, total_sample_count),
                              file=file_eval_best)
                        file_eval_best.close()

                        sourceFile = detail_filename
                        targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_detail')
                        if os.path.exists(targetFile):
                            os.remove(targetFile)
                        shutil.copy(sourceFile, targetFile)
                        sourceFile = ckpt.model_checkpoint_path
                        targetFile = os.path.join(FLAGS.eval_dir, 'best_eval.ckpt')
                        if os.path.exists(targetFile):
                            os.remove(targetFile)
                        shutil.copy(sourceFile, targetFile)
                    best_eval_ckpt = global_step_for_restore


def main(argv=None):  # pylint: disable=unused-argument
    global evalDataSet
    # assert not tf.gfile.Exists(FLAGS.eval_dir), 'please move the old evaluate directory to pre_versions!'
    if tf.gfile.Exists(FLAGS.eval_dir):
        print('the evaluate data has already exists!')
        str = input('continue will delete the old evaluate directory:(y/n)')
        if str == 'y' or str == 'Y':
            tf.gfile.DeleteRecursively(FLAGS.eval_dir)
        elif str == 'n' or str == 'N':
            print('eval end!')
            return
        else:
            print('invalid input!')
            return
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    # checkpoint = input('please input the choosed checkpoint to eval:(0 for latest)')
    checkpoint = '0'
    evalDataSet = graphcnn_input.generate_eval_data(EVAL_DATA_DIR,ont_hot=False)
    evaluate(checkpoint)


if __name__ == '__main__':
    tf.app.run()

