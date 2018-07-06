
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

evalDataSet = None

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tmp/graphcnn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/graphcnn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


   
def evaluate(checkpoint):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        data = tf.placeholder(tf.float32, [graphcnn_input.EVAL_BATCH_SIZE, graphcnn_input.NUM_CHANNELS])  # NUM_CHANNELS: feature dim
        labels = tf.placeholder(tf.int32, [graphcnn_input.EVAL_BATCH_SIZE,graphcnn_input.NUM_CLASSES])  # with value: -1,1

        # inference model.
        logits = SVM_model.SVM_inference(data, labels, dependencies_loss=False)
        
        # Declare prediction function
        prediction = tf.sign(logits)
        
        # Restore the moving average version of the learned variables for eval. # ?????????????????????????
        variable_averages = tf.train.ExponentialMovingAverage(graphcnn_option.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        # summary_op = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)
        
        
        best_eval_value = 0;  ####
        best_eval_ckpt = 0;  ####
        
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
            total_true_value = np.zeros([1, graphcnn_input.NUM_CLASSES], dtype=np.int32)
            while step < num_iter:
                test_data, test_label = evalDataSet.next_batch(graphcnn_input.EVAL_BATCH_SIZE)
                start_time = time.time()
                predicted_value, true_value = sess.run(
                    [prediction, labels], feed_dict={data: test_data, labels: test_label})
                duration = time.time() - start_time
                sec_per_batch = float(duration)
                print('sec_per_batch:%.3f/%d'%(sec_per_batch,graphcnn_input.EVAL_BATCH_SIZE))
                total_predicted_value = np.concatenate((total_predicted_value, predicted_value), axis=0)
                total_true_value = np.concatenate((total_true_value, true_value), axis=0)
                step += 1

            total_predicted_value = total_predicted_value[1:]
            total_true_value = total_true_value[1:]
            
            detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution')
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_predicted_value, fmt='%.4f')
            total_predicted_value = ((total_predicted_value) >= graphcnn_option.EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)
            assert total_sample_count == total_predicted_value.shape[0], 'sample_count error!'
            detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_predicted_value, fmt='%d')
            detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_true_value, fmt='%d')


            filename_eval_log = os.path.join(FLAGS.eval_dir, 'log_eval')
            file_eval_log = open(filename_eval_log, 'w')
            np.set_printoptions(threshold=np.nan)
            print('\nevaluation:', file=file_eval_log)
            print('\nevaluation:')
            print('  %s, ckpt-%d:' % (datetime.now(), global_step_for_restore), file=file_eval_log)
            print('  %s, ckpt-%d:' % (datetime.now(), global_step_for_restore))

            total_predicted_value = total_predicted_value.astype(bool)
            total_true_value = total_true_value.astype(bool)

            print('  example based evaluations:', file=file_eval_log)
            print('  example based evaluations:')

            equal = total_true_value == total_predicted_value
            match = np.sum(equal, axis=1) == np.size(equal, axis=1)
            exact_match_ratio = np.sum(match) / np.size(match)
            print('      exact_match_ratio = %.4f' % exact_match_ratio, file=file_eval_log)
            print('      exact_match_ratio = %.4f' % exact_match_ratio)

            true_and_predict = np.sum(total_true_value & total_predicted_value, axis=1)
            true_or_predict = np.sum(total_true_value | total_predicted_value, axis=1)
            accuracy = np.mean(true_and_predict / true_or_predict)
            print('      accuracy = %.4f' % accuracy, file=file_eval_log)
            print('      accuracy = %.4f' % accuracy)

            precison = np.mean(true_and_predict / (np.sum(total_predicted_value, axis=1) + 1e-9))
            print('      precison = %.4f' % precison, file=file_eval_log)
            print('      precison = %.4f' % precison)

            recall = np.mean(true_and_predict / np.sum(total_true_value, axis=1))
            print('      recall = %.4f' % recall, file=file_eval_log)
            print('      recall = %.4f' % recall)

            F1_Measure = np.mean((true_and_predict * 2) / (np.sum(total_true_value, axis=1)
                                                           + np.sum(total_predicted_value, axis=1)))
            print('      F1_Measure = %.4f' % F1_Measure, file=file_eval_log)
            print('      F1_Measure = %.4f' % F1_Measure)

            HammingLoss = np.mean(total_true_value ^ total_predicted_value)
            print('      HammingLoss = %.4f' % HammingLoss, file=file_eval_log)
            print('      HammingLoss = %.4f' % HammingLoss)


            print('  label based evaluations:', file=file_eval_log)
            print('  label based evaluations:')

            TP = np.sum(total_true_value & total_predicted_value,axis=0,dtype=np.int32)
            FP = np.sum((~total_true_value) & total_predicted_value,axis=0,dtype=np.int32)
            FN = np.sum(total_true_value & (~total_predicted_value),axis=0,dtype=np.int32)

            _P = np.sum(TP) / (np.sum(TP) + np.sum(FP)  + 1e-9 )
            _R = np.sum(TP) / (np.sum(TP) + np.sum(FN)  + 1e-9 )
            Micro_F1 = (2 * _P *_R) / (_P + _R)
            print('      P = %.4f' % _P, file=file_eval_log)
            print('      P = %.4f' % _P)
            print('      R = %.4f' % _R, file=file_eval_log)
            print('      R = %.4f' % _R)
            print('      Micro-F1 = %.4f' % Micro_F1, file=file_eval_log)
            print('      Micro-F1 = %.4f' % Micro_F1)

            _P_t = TP / (TP + FP + 1e-9)
            _R_t = TP / (TP + FN + 1e-9)
            Macro_F1 = np.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-9))
            # print('    P_t = %.4f' % _P, file=file_eval_log)
            # print('    P_t = %.4f' % _P)
            # print('    R_t = %.4f' % _R, file=file_eval_log)
            # print('    R_t = %.4f' % _R)
            print('      Macro-F1 = %.4f' % Macro_F1, file=file_eval_log)
            print('      Macro-F1 = %.4f' % Macro_F1)


            print('evaluation samples number:%d, evaluation classes number:%d' %
                  (total_predicted_value.shape[0], total_predicted_value.shape[1]), file=file_eval_log)
            print('evaluation samples number:%d, evaluation classes number:%d' %
                  (total_predicted_value.shape[0], total_predicted_value.shape[1]))
            # print('evaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
            #       + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
            #       + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution'),
            #       file=file_eval_log)
            # print('evaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval')
            #       + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
            #       + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
            #       + ', ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution'))
            file_eval_log.close()

            best_eval_ckpt = global_step_for_restore
            best_eval_value = Macro_F1
            sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval')
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)
            sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_true_value')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)
            sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_predicted_value')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)
            sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution')
            targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_predicted_value_dictribution')
            if os.path.exists(targetFile):
                os.remove(targetFile)
            shutil.copy(sourceFile, targetFile)
            # sourceFile = ckpt.model_checkpoint_path + '.index'
            # targetFile = os.path.join(FLAGS.eval_dir, 'best_eval.ckpt')
            # if os.path.exists(targetFile):
            #     os.remove(targetFile)
            # shutil.copy(sourceFile, targetFile)

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
                    total_sample_count = num_iter * graphcnn_input.EVAL_BATCH_SIZE
                    step = 0
                    total_predicted_value = np.zeros([1,graphcnn_input.NUM_CLASSES],dtype=np.float32) ##
                    total_true_value = np.zeros([1,graphcnn_input.NUM_CLASSES],dtype=np.int32)
                    while step < num_iter:
                        test_data, test_label = evalDataSet.next_batch(graphcnn_input.EVAL_BATCH_SIZE)
                        predicted_value, true_value = sess.run(
                            [prediction, labels], feed_dict={data: test_data, labels: test_label})
                        total_predicted_value = np.concatenate((total_predicted_value,predicted_value),axis=0)
                        total_true_value = np.concatenate((total_true_value, true_value),axis=0)
                        step += 1

                    total_predicted_value = total_predicted_value[1:]
                    total_true_value = total_true_value[1:]

                    detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution')
                    if os.path.exists(detail_filename):
                        os.remove(detail_filename)
                    np.savetxt(detail_filename, total_predicted_value, fmt='%.4f')
                    total_predicted_value = ((total_predicted_value)>= graphcnn_option.EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)
                    assert total_sample_count == total_predicted_value.shape[0], 'sample_count error!'
                    detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
                    if os.path.exists(detail_filename):
                        os.remove(detail_filename)
                    np.savetxt(detail_filename, total_predicted_value, fmt='%d')
                    detail_filename = os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
                    if os.path.exists(detail_filename):
                        os.remove(detail_filename)
                    np.savetxt(detail_filename, total_true_value, fmt='%d')

                    filename_eval_log = os.path.join(FLAGS.eval_dir, 'log_eval')
                    file_eval_log = open(filename_eval_log, 'a')
                    np.set_printoptions(threshold=np.nan)
                    print('\nevaluation:', file=file_eval_log)
                    print('\nevaluation:')
                    print('  %s, ckpt-%d:' % (datetime.now(),global_step_for_restore), file=file_eval_log)
                    print('  %s, ckpt-%d:' % (datetime.now(),global_step_for_restore))

                    total_predicted_value = total_predicted_value.astype(bool)
                    total_true_value = total_true_value.astype(bool)

                    print('  example based evaluations:', file=file_eval_log)
                    print('  example based evaluations:')

                    equal = total_true_value == total_predicted_value
                    match = np.sum(equal, axis=1) == np.size(equal, axis=1)
                    exact_match_ratio = np.sum(match) / np.size(match)
                    print('      exact_match_ratio = %.4f' % exact_match_ratio, file=file_eval_log)
                    print('      exact_match_ratio = %.4f' % exact_match_ratio)

                    true_and_predict = np.sum(total_true_value & total_predicted_value, axis=1)
                    true_or_predict = np.sum(total_true_value | total_predicted_value, axis=1)
                    accuracy = np.mean(true_and_predict / true_or_predict)
                    print('      accuracy = %.4f' % accuracy, file=file_eval_log)
                    print('      accuracy = %.4f' % accuracy)

                    precison = np.mean(true_and_predict / (np.sum(total_predicted_value, axis=1)+1e-9))
                    print('      precison = %.4f' % precison, file=file_eval_log)
                    print('      precison = %.4f' % precison)

                    recall = np.mean(true_and_predict / np.sum(total_true_value, axis=1))
                    print('      recall = %.4f' % recall, file=file_eval_log)
                    print('      recall = %.4f' % recall)

                    F1_Measure = np.mean( (true_and_predict*2) / (np.sum(total_true_value, axis=1)
                                                                  +np.sum(total_predicted_value, axis=1)))
                    print('      F1_Measure = %.4f' % F1_Measure, file=file_eval_log)
                    print('      F1_Measure = %.4f' % F1_Measure)

                    HammingLoss = np.mean(total_true_value ^ total_predicted_value)
                    print('      HammingLoss = %.4f' % HammingLoss, file=file_eval_log)
                    print('      HammingLoss = %.4f' % HammingLoss)

                    print('  label based evaluations:', file=file_eval_log)
                    print('  label based evaluations:')


                    TP = np.sum(total_true_value & total_predicted_value, axis=0, dtype=np.int32)
                    FP = np.sum((~total_true_value) & total_predicted_value, axis=0, dtype=np.int32)
                    FN = np.sum(total_true_value & (~total_predicted_value), axis=0, dtype=np.int32)

                    _P = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-9)
                    _R = np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-9)
                    Micro_F1 = (2 * _P * _R) / (_P + _R + 1e-9)
                    print('      P = %.4f' % _P, file=file_eval_log)
                    print('      P = %.4f' % _P)
                    print('      R = %.4f' % _R, file=file_eval_log)
                    print('      R = %.4f' % _R)
                    print('      Micro-F1 = %.4f' % Micro_F1, file=file_eval_log)
                    print('      Micro-F1 = %.4f' % Micro_F1)

                    _P_t = TP / (TP + FP + 1e-9)
                    _R_t = TP / (TP + FN + 1e-9)
                    # assert _P_t.shape[0]==graphcnn_input.NUM_CLASSES, '_P_t has a wrong size'
                    Macro_F1 = np.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-9))
                    # print('    P_t = %.4f' % _P, file=file_eval_log)
                    # print('    P_t = %.4f' % _P)
                    # print('    R_t = %.4f' % _R, file=file_eval_log)
                    # print('    R_t = %.4f' % _R)
                    print('      Macro-F1 = %.4f' % Macro_F1, file=file_eval_log)
                    print('      Macro-F1 = %.4f' % Macro_F1)

                    print('evaluation samples number:%d, evaluation classes number:%d' %
                          (total_predicted_value.shape[0],total_predicted_value.shape[1]),file=file_eval_log)
                    print('evaluation samples number:%d, evaluation classes number:%d' %
                          (total_predicted_value.shape[0], total_predicted_value.shape[1]))
                    # print('evaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
                    #       + ', '+os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value'),
                    #       file=file_eval_log)
                    # print('evaluation detail: ' + os.path.join(FLAGS.eval_dir, 'log_eval')
                    #       + ', '+ os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
                    #       + ', '+ os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
                    #       + ', '+ os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution'))

                    file_eval_log.close()

                    if accuracy > best_eval_value:
                        best_eval_value = Macro_F1

                        filename_eval_best = os.path.join(FLAGS.eval_dir, 'best_eval')
                        file_eval_best = open(filename_eval_best, 'w')
                        print('evaluation:', file=file_eval_best)
                        print('  %s, ckpt-%d:' % (datetime.now(), global_step_for_restore), file=file_eval_best)
                        print('  example based evaluations:', file=file_eval_best)
                        print('      exact_match_ratio = %.4f' % exact_match_ratio, file=file_eval_best)
                        print('      accuracy = %.4f' % accuracy, file=file_eval_best)
                        print('      precison = %.4f' % precison, file=file_eval_best)
                        print('      recall = %.4f' % recall, file=file_eval_best)
                        print('      F1_Measure = %.4f' % F1_Measure, file=file_eval_best)
                        print('      HammingLoss = %.4f' % HammingLoss, file=file_eval_best)
                        print('  label based evaluations:', file=file_eval_best)
                        print('      P = %.4f' % _P, file=file_eval_best)
                        print('      R = %.4f' % _R, file=file_eval_best)
                        print('      Micro-F1 = %.4f' % Micro_F1, file=file_eval_best)
                        print('      Macro-F1 = %.4f' % Macro_F1, file=file_eval_best)
                        print('evaluation samples number:%d, evaluation classes number:%d' %
                              (total_predicted_value.shape[0], total_predicted_value.shape[1]), file=file_eval_best)
                        print('evaluation detail: ' + os.path.join(FLAGS.eval_dir, 'best_eval_for_true_value')
                              + ', ' + os.path.join(FLAGS.eval_dir, 'best_eval_for_predicted_value')
                              + ', ' + os.path.join(FLAGS.eval_dir, 'best_eval_for_predicted_value_dictribution'),
                              file=file_eval_best)
                        file_eval_best.close()

                        sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_true_value')
                        targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_true_value')
                        if os.path.exists(targetFile):
                            os.remove(targetFile)
                        shutil.copy(sourceFile, targetFile)
                        sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value')
                        targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_predicted_value')
                        if os.path.exists(targetFile):
                            os.remove(targetFile)
                        shutil.copy(sourceFile, targetFile)
                        sourceFile = os.path.join(FLAGS.eval_dir, 'log_eval_for_predicted_value_dictribution')
                        targetFile = os.path.join(FLAGS.eval_dir, 'best_eval_for_predicted_value_dictribution')
                        if os.path.exists(targetFile):
                            os.remove(targetFile)
                        shutil.copy(sourceFile, targetFile)
                        # sourceFile = ckpt.model_checkpoint_path
                        # targetFile = os.path.join(FLAGS.eval_dir, 'best_eval.ckpt')
                        # if os.path.exists(targetFile):
                        #     os.remove(targetFile)
                        # shutil.copy(sourceFile, targetFile)
                    best_eval_ckpt = global_step_for_restore

                    # summary = tf.Summary()
                    # summary.ParseFromString(sess.run(summary_op))
                    # summary.value.add(tag='Precision @ 1', simple_value=precision)
                    # summary_writer.add_summary(summary, global_step_for_restore)
        
        
        
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

    # evalDataSet = graphcnn_input.generate_eval_data(graphcnn_option.EVAL_DATA_DIR,
    #                                                 shuffled = False,
    #                                                 ont_hot=True,
    #                                                 index_mode=True)

    # 可以通过调用train data来用train数据进行测试：
    evalDataSet = graphcnn_input.generate_train_data(graphcnn_option.EVAL_DATA_DIR,
                                                     shuffled = False,
                                                     ont_hot=True,
                                                     index_mode=True)

    print('evaluating...')
    evaluate(checkpoint)
      

if __name__ == '__main__':
    tf.app.run()







