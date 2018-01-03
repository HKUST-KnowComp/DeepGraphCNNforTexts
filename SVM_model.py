
# HR-SVM

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import math

import graphcnn_input
import graphcnn_option



class Model(object):
    ''' svm model
    '''
    
    def __init__(self):
        self._paramaters_list = []
       
    def linear_SVM(self, data, target):
    ''' Linear Support Vector Machine: Soft Margin
        data: 2D of [samples number, feature vector dimension]
        target: 2D of [samples number, 1], with value -1 or 1
    '''
        # feature vector dimension
        feature_dim = data.get_shape()[1].value
        
        # Create variables for linear regression
        A = tf.Variable(tf.random_normal(shape=[feature_dim,1]))
        b = tf.Variable(tf.random_normal(shape=[1,1]))
        
        # record para
        self._paramaters_list.append(A)
        self._paramaters_list.append(b)
        
        # Declare model operations
        model_output = tf.sub(tf.matmul(data, A), b)
        
        # Declare vector L2 'norm' function squared
        l2_norm = tf.reduce_sum(tf.square(A))

        # Declare loss function
        # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
        # L2 regularization parameter, alpha
        alpha = tf.constant([0.01])
        # Margin term in loss
        classification_term = tf.reduce_mean(tf.maximum(0., tf.sub(1., tf.mul(model_output, target))))
        # Put terms together
        loss = tf.add(classification_term, tf.mul(alpha, l2_norm),name='svm_loss')
        
        tf.add_to_collection('losses', loss)
        
        return model_output


def compute_dependencies_loss(model_list):
    # Calculate the Variable's dependency constraint
    filename = os.path.join(graphcnn_option.DATA_PATH, 'fathercode')
    father = np.loadtxt(filename, dtype=int)

    # Calculate the inner nodes' parameters value
    inner = np.zeros([graphcnn_input.NUM_CLASSES])
    for i in range(0, graphcnn_input.NUM_CLASSES):
        father_i = father[i]
        if father_i != -1:
            inner[father_i] = 1
    nodes = []
    for i in range(0, graphcnn_input.NUM_CLASSES):
        nodes.append([])
    for i in range(0, graphcnn_input.NUM_CLASSES):
        if inner[i] == 1:
            father_i = father[i]
            nodes[i].append(model_list[i]._paramaters_list)
            if father_i != -1:
                nodes[i].append(model_list[father_i]._paramaters_list)
                nodes[father_i].append(model_list[i]._paramaters_list)
    nodes_paras = []
    for i in range(0, graphcnn_input.NUM_CLASSES):
        para_list = []
        if inner[i] == 1:
            para_list_len = len(nodes[i][0])
            para_list_num = len(nodes[i])
            for para_i in range(0,para_list_len):
                para = []
                for para_list_i in range(0,para_list_num):
                    para.append(nodes[i][para_list_i][para_i])
                para_list.append(tf.truediv(tf.add_n(para), float(para_list_num))) ##???????????????
        nodes_paras.append(para_list)

    for i in range(0, graphcnn_input.NUM_CLASSES):
        if inner[i] == 1:
            model_para = model_list[i]._paramaters_list
            father_model_para = nodes_paras[i]
        else:
            model_para = model_list[i]._paramaters_list
            father_i = father[i]
            if father_i != -1:
                father_model_para = nodes_paras[father_i]
        assert len(model_para) == len(father_model_para), ' something is wrong'
        for j in range(0, len(model_para)):
            sub_vector = tf.sub(model_para[j], father_model_para[j])
            reshape = tf.reshape(sub_vector, [1, -1])
            reshape_trans = tf.reshape(sub_vector, [-1, 1])
            dependencies = tf.mul(tf.matmul(reshape, reshape_trans)[0, 0], graphcnn_option.VARIABLE_DEPENDENCY,
                                  name='dependencies_loss')
            tf.add_to_collection('losses', dependencies)

def SVM_inference(data, target, dependencies_loss=True):
    '''
        data: 2D of [samples number, feature vector dimension]
        target: 2D of [samples number, NUM_CLASSES], with value -1 or 1
    '''
    
    model_list = []
    logits_list = []
    for i in range(0, graphcnn_input.NUM_CLASSES):
        target_i = target[:,i]
        target_i = tf.reshape(target_i, [-1, 1])
        model = Model()
        logits = model.linear_SVM(data, target_i)
        model_list.append(model)
        logits_list.append(logits)
    logits = tf.concat(1, logits_list)

    if dependencies_loss:
        compute_dependencies_loss(model_list)

    return logits     

def SVM_loss():
    ''' add loss function: cross entropy.
    '''
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
def _add_loss_summaries(total_loss):
    """ Add summaries for losses.
        Generates moving average for all losses and associated summaries for visualizing the performance of the network.
        moving average -> eliminate noise

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    # The moving averages are computed using exponential decay:
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)   equivalent to:
    # shadow_variable = decay * shadow_variable + (1 - decay) * variable
    loss_averages = tf.train.ExponentialMovingAverage(graphcnn_option.MOVING_AVERAGE_DECAY, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    if graphcnn_option.SUMMARYWRITER:
        # Attach a scalar summary to all individual losses and the total loss; do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

def SVM_train(total_loss, global_step):
    """ Create an optimizer and apply to all trainable variables.
        Add moving average for all trainable variables.

    Args:
        total_loss: total loss from loss().
        global_step: Integer Variable counting the number of training steps processed.

    Returns:
        train_op: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = graphcnn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / graphcnn_input.TRAIN_BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * graphcnn_option.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    # decayed_learning_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_RATE ^ (global_step / decay_steps)
    lr = tf.train.exponential_decay(graphcnn_option.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    graphcnn_option.LEARNING_RATE_DECAY_RATE,
                                    staircase=True)

    if graphcnn_option.SUMMARYWRITER:
        tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, graphcnn_option.MOMENTUM)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if graphcnn_option.SUMMARYWRITER:
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        graphcnn_option.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op





