
# 20

import tensorflow as tf
import graphcnn_input
import graphcnn_option
import os.path
import numpy as np

class Model(object):
    ''' graph cnn model
    '''

    def __init__(self):
        self._paramaters_list = []

    def _activation_summary(self, x):
        """ Helper to create summaries for activations.
            Creates a summary that provides a histogram of activations.
            Creates a summary that measures the sparsity of activations.

        Args:
            x: Tensor
        Returns:
            nothing
        """
        if graphcnn_option.SUMMARYWRITER:
            tensor_name = x.op.name
            tf.histogram_summary(tensor_name + '/activations', x)  # Outputs a Summary protocol buffer with a histogram.
            tf.scalar_summary(tensor_name + '/sparsity',
                              tf.nn.zero_fraction(x))  # Outputs a Summary protocol buffer with scalar values.

    def _variable_with_weight_decay(self, name, initial_value, wd=None):
        """ Helper to create an initialized Variable with weight decay.
            Note that the Variable is initialized with a truncated normal distribution.
            A weight decay is added only if one is specified.

        Args:
            name: name of the variable
            initial_value: initial value for Variable
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

        Returns:
            Variable Tensor
        """
        dtype = tf.float16 if graphcnn_option.USE_FP16 else tf.float32
        var = tf.Variable(initial_value=initial_value, name=name, dtype=dtype)
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name = name + '_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    ## RCV1, graph cnn, 3 conv layers
    def inference(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)



        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 64
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool1  (max pooling, cross pooling)
        with tf.variable_scope('pool1') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                     padding='VALID', name=scope.name)


        # conv2
        with tf.variable_scope('conv2') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 256 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 96  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        return softmax_linear

    def inference_head(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # with tf.variable_scope('head'):
        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 64
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool1  (max pooling, cross pooling)
        with tf.variable_scope('pool1') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                     padding='VALID', name=scope.name)

        return output

    def inference_end(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # conv2
        with tf.variable_scope('conv2') as scope:
            inputmaps = input.get_shape()[3].value
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 256 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 96  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        return softmax_linear


    ## RCV1, graph cnn, 1 conv layers
    def inference_1conv_head(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # with tf.variable_scope('head'):
        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 64 # 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)

        return output

    def inference_1conv_end(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(input, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 512 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 64  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        return softmax_linear


    ## RCV1, graph cnn, 6 conv layers
    def inference_6conv_head(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # with tf.variable_scope('head'):
        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 64
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)

        # conv2
        with tf.variable_scope('conv2') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)

        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv4
        with tf.variable_scope('conv4') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 256
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm4 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm4') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)

        # conv5
        with tf.variable_scope('conv5') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 256
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm5 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm5') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)

        # conv6
        with tf.variable_scope('conv6') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 256
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm6 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm6') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)

        return output

    def inference_6conv_end(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)

        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(input, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 512 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 64  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)

        return softmax_linear


    ## LSTHC, graph cnn, 3 conv layers
    def LSHTC_3conv_inference(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)



        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 192
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)


        # conv2
        with tf.variable_scope('conv2') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 512
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool2  (max pooling, cross pooling)
        # with tf.variable_scope('pool2') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                                 padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 1024
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool3 (max pooling, cross pooling)
        # with tf.variable_scope('pool3') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                                 padding='VALID', name=scope.name)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = graphcnn_input.NUM_CLASSES  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)

        return softmax_linear

    def LSHTC_3conv_inference_head(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # with tf.variable_scope('head'):
        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 64
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool1  (max pooling, cross pooling)
        with tf.variable_scope('pool1') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                     padding='VALID', name=scope.name)

        return output

    def LSHTC_3conv_inference_end(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # conv2
        with tf.variable_scope('conv2') as scope:
            inputmaps = input.get_shape()[3].value
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # record para
        self._paramaters_list.append(weights)
        self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 256 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 96  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        return softmax_linear



    ## LSTHC, graph cnn, 6 conv layers
    # 316*5 - 316 - 312(156) - 152(76) - 72(36) - 32(16) - 12  -- 65536
    # 50    - 128 - 256      - 512     - 1024   - 2048   - 4096
    def LSHTC_6conv_inference(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)



        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)


        # conv2
        with tf.variable_scope('conv2') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 256
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 512
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)

        # conv4
        with tf.variable_scope('conv4') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 1024
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm4 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm4') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool4 (max pooling, cross pooling)
        with tf.variable_scope('pool4') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv5
        with tf.variable_scope('conv5') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 2048
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm5 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm5') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool5 (max pooling, cross pooling)
        with tf.variable_scope('pool5') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv6
        with tf.variable_scope('conv6') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 4096
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm6 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm6') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool6 (max pooling, cross pooling)
        # with tf.variable_scope('pool6') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                             padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 65536 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # # fc2 (fully connected layers)
        # with tf.variable_scope('fc2') as scope:
        #     input = output
        #     inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
        #     outputmaps = 96  # neuturons numbers
        #     weights = self._variable_with_weight_decay('weights',
        #                                                    tf.truncated_normal(shape=[inputmaps, outputmaps],
        #                                                                        stddev=0.04),
        #                                                    wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
        #     biases = self._variable_with_weight_decay('biases',
        #                                                   tf.constant(0.1, shape=[outputmaps]),
        #                                                   wd=graphcnn_option.WEIGHT_DECAY)
        #     fc = tf.matmul(input, weights)
        #     fc = tf.nn.bias_add(fc, biases)
        #     output = tf.nn.relu(fc, name=scope.name)
        #     self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        #
        # # Dropout also scales activations such that no rescaling is needed at evaluation time.
        # if not eval_data:
        #     output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        return softmax_linear

    def LSHTC_5conv_inference_head(self, input, eval_data=False):
        """
        # 92*5 - 92
        # 50   - 96
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # with tf.variable_scope('head'):
        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)

        return output

    def LSHTC_5conv_inference_end(self, input, eval_data=False):
        """
        # 92 - 88(44) - 40(20) - 16(8) - 4(2)  -- 128 -- 32 -- 1
        # 96 - 128    - 128    - 256   - 256
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # conv2
        with tf.variable_scope('conv2') as scope:
            inputmaps = input.get_shape()[3].value
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)

        # conv4
        with tf.variable_scope('conv4') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 256
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm4 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm4') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool4 (max pooling, cross pooling)
        with tf.variable_scope('pool4') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv5
        with tf.variable_scope('conv5') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 256
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm5 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm5') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool5 (max pooling, cross pooling)
        with tf.variable_scope('pool5') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 128 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 32  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        return softmax_linear


    ## LSTHC, graph cnn, 5 conv layers
    # 192*5 - 192 - 188(94) - 90(45) - 41(20) - 16  -- 9216 - 4096 - 4096(2048) - N
    # 50    - 128 - 128     - 192    -320     - 576
    def LSHTC_hier_6conv_inference(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        # if eval_data:
        #     batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        # else:
        #     batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)

        batch_size = input.get_shape()[0].value



        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)


        # conv2
        with tf.variable_scope('conv2') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 192
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)

        # conv4
        with tf.variable_scope('conv4') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 320
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm4 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm4') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool4 (max pooling, cross pooling)
        with tf.variable_scope('pool4') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv5
        with tf.variable_scope('conv5') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 576
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm5 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm5') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool5 (max pooling, cross pooling)
        # with tf.variable_scope('pool5') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                             padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 4096 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 4096 # 2048  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)

        eigenvectors = output

        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        return softmax_linear, eigenvectors

    def LSHTC_hier_6conv_inference_head(self, input, eval_data=False):
        """
        # 192*5 - 192
        # 50   - 64
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # with tf.variable_scope('head'):
        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 64
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)

        return output

    def LSHTC_hier_6conv_inference_end(self, input, eval_data=False):
        """
        # 192 - 188(94) - 90(45) - 41(20) - 16(8) - 4  -- 896 - 256 - 64 - 1
        # 64  - 96      - 128    - 160    - 192   - 224
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)


        # conv2
        with tf.variable_scope('conv2') as scope:
            inputmaps = input.get_shape()[3].value
            outputmaps = 96
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)

        # conv4
        with tf.variable_scope('conv4') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 160
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm4 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm4') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool4 (max pooling, cross pooling)
        with tf.variable_scope('pool4') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv5
        with tf.variable_scope('conv5') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 192
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm5 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm5') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool5 (max pooling, cross pooling)
        with tf.variable_scope('pool5') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv6
        with tf.variable_scope('conv6') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 224
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm6 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm6') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool6 (max pooling, cross pooling)
        # with tf.variable_scope('pool6') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                             padding='VALID', name=scope.name)

        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 256 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 64  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 1 # graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        return softmax_linear


    ## LSTHC, graph cnn, 5 conv layers
    # 192*5 - 192 - 188(94) - 90(45) - 41(20) - 16  -- 9216 - 4096 - 4096(2048) - N
    # 50    - 128 - 128     - 192    -320     - 576
    def NYT_5conv_inference(self, input, eval_data=False):
        """
        Args:
            input: 4D tensor of [batch_size, WIDTH, HEIGHT, DEPTHS] size.
        Returns:
            logits: 2D tensor of [batch_size, NUM_CLASSES].
        """

        if eval_data:
            batch_size = int(graphcnn_input.EVAL_BATCH_SIZE)
        else:
            batch_size = int(graphcnn_input.TRAIN_BATCH_SIZE)



        # conv1
        with tf.variable_scope('conv1') as scope:
            inputmaps = graphcnn_input.NUM_CHANNELS
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.0, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 5, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm1 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm1') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool1  (max pooling, cross pooling)
        # with tf.variable_scope('pool1') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                              padding='VALID', name=scope.name)


        # conv2
        with tf.variable_scope('conv2') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 128
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm2 (normalization)  tf.nn.local_response_normalization
        with tf.variable_scope('norm2') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool2  (max pooling, cross pooling)
        with tf.variable_scope('pool2') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)


        # conv3
        with tf.variable_scope('conv3') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 192
            weights = self._variable_with_weight_decay('weights',
                                                             tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                                 stddev=5e-2),
                                                             wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                            tf.constant(0.1, shape=[outputmaps]),
                                                            wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)
        # norm3 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm3') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool3 (max pooling, cross pooling)
        with tf.variable_scope('pool3') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                        padding='VALID', name=scope.name)

        # conv4
        with tf.variable_scope('conv4') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 320
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm4 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm4') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # pool4 (max pooling, cross pooling)
        with tf.variable_scope('pool4') as scope:
            output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                                    padding='VALID', name=scope.name)

        # conv5
        with tf.variable_scope('conv5') as scope:
            input = output
            inputmaps = outputmaps
            outputmaps = 576
            weights = self._variable_with_weight_decay('weights',
                                                       tf.truncated_normal(shape=[1, 5, inputmaps, outputmaps],
                                                                           stddev=5e-2),
                                                       wd=graphcnn_option.WEIGHT_DECAY)
            biases = self._variable_with_weight_decay('biases',
                                                      tf.constant(0.1, shape=[outputmaps]),
                                                      wd=graphcnn_option.WEIGHT_DECAY)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(conv, name=scope.name)
            self._activation_summary(output)
        # norm5 (normalization) tf.nn.local_response_normalization
        with tf.variable_scope('norm5') as scope:
            output = tf.nn.lrn(output, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                               name=scope.name)
        # # pool5 (max pooling, cross pooling)
        # with tf.variable_scope('pool5') as scope:
        #     output = tf.nn.max_pool(output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
        #                             padding='VALID', name=scope.name)


        # fc1 (fully connected layers)
        with tf.variable_scope('fc1') as scope:
            input = tf.reshape(output, [batch_size, -1])
            inputmaps = input.get_shape()[1].value
            outputmaps = 7168 #6144 # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                  tf.truncated_normal(shape=[inputmaps, outputmaps], stddev=0.04),
                                                  wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                 tf.constant(0.1, shape=[outputmaps]),
                                                 wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc2 (fully connected layers)
        with tf.variable_scope('fc2') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = 7168 #4096 # 2048  # neuturons numbers
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev=0.04),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.1, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            fc = tf.nn.bias_add(fc, biases)
            output = tf.nn.relu(fc, name=scope.name)
            self._activation_summary(output)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        # Dropout also scales activations such that no rescaling is needed at evaluation time.
        if not eval_data:
            output = tf.nn.dropout(output, graphcnn_option.DROPOUT_FRACTION)


        # fc3 (softmax, i.e. softmax(WX + b))
        with tf.variable_scope('softmax_linear') as scope:
            input = output
            inputmaps = outputmaps  # inputmaps = input.get_shape()[1].value
            outputmaps = graphcnn_input.NUM_CLASSES
            weights = self._variable_with_weight_decay('weights',
                                                           tf.truncated_normal(shape=[inputmaps, outputmaps],
                                                                               stddev= 1.0 / inputmaps),
                                                           wd=graphcnn_option.WEIGHT_DECAY)  # ????????????????????????
            biases = self._variable_with_weight_decay('biases',
                                                          tf.constant(0.0, shape=[outputmaps]),
                                                          wd=graphcnn_option.WEIGHT_DECAY)
            fc = tf.matmul(input, weights)
            softmax_linear = tf.nn.bias_add(fc, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        # # record para
        # self._paramaters_list.append(weights)
        # self._paramaters_list.append(biases)

        return softmax_linear





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


def inference_GPU(data, eval_data=False, dependencies_loss=True):
    # inference model.
    model_list = []
    logits_list = []
    num_model_per_GPU = int(graphcnn_input.NUM_CLASSES / graphcnn_option.NUM_GPUS)
    for i in range(graphcnn_option.NUM_GPUS - 1):
        devicename = '/cpu:0' if eval_data else '/gpu:%d' % i
        with tf.device(devicename):
            with tf.name_scope('%s_%d' % (graphcnn_option.TOWER_NAME, i)) as scope:
                for j in range(num_model_per_GPU * i, num_model_per_GPU * (i + 1)):
                    model = Model()
                    logits = model.inference(data, eval_data=eval_data)
                    model_list.append(model)
                    logits_list.append(logits)
    devicename = '/cpu:0' if eval_data else '/gpu:%d' % (graphcnn_option.NUM_GPUS - 1)
    with tf.device(devicename):
        with tf.name_scope('%s_%d' % (graphcnn_option.TOWER_NAME, (graphcnn_option.NUM_GPUS - 1))) as scope:
            for j in range(num_model_per_GPU * (graphcnn_option.NUM_GPUS - 1), graphcnn_input.NUM_CLASSES):
                model = Model()
                logits = model.inference(data, eval_data = eval_data)
                model_list.append(model)
                logits_list.append(logits)
    logits = tf.concat(1, logits_list)

    if dependencies_loss:
        compute_dependencies_loss(model_list)

    return logits


def inference_CPU(data, eval_data=False, dependencies_loss=True):
    # inference model.
    model = Model()
    data = model.LSHTC_hier_6conv_inference_head(data, eval_data=eval_data)

    model_list = []
    logits_list = []
    for i in range(0, graphcnn_input.NUM_CLASSES):
        model = Model()
        logits = model.LSHTC_hier_6conv_inference_end(data, eval_data=eval_data)
        model_list.append(model)
        logits_list.append(logits)
    logits = tf.concat(1, logits_list)

    if dependencies_loss:
        compute_dependencies_loss(model_list)

    return logits


def inference(data, eval_data=False, eigenvectors = False):
    model = Model()
    logits, vectors = model.LSHTC_hier_6conv_inference(data, eval_data=eval_data)
    if eigenvectors:
        return logits, vectors
    else:
        return logits

    # return inference_CPU(data, eval_data=eval_data,dependencies_loss=False)




def loss(logits, labels):
    """ add loss function: cross entropy.

    Args:
        logits: Logits from inference(), 2D tensor of [batch_size, NUM_CLASSES].
        labels: 2D tensor of [batch_size, NUM_CLASSES].

    Returns:
        Loss: 0D tensor of type float.
    """

    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits, labels, name='cross_entropy_per_example')   # shape = [batch_size]

    # single label
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     logits, labels, name='cross_entropy_per_example')   # shape = [batch_size]

    # multi labels
    # logits = tf.cast(logits, tf.float64)
    # labels = tf.cast(labels, tf.float64)
    labels = tf.cast(labels, logits.dtype)
    sigmoid_cross_entropy_per_example = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels,
                                                                                name='sigmoid_cross_entropy_per_example')
    cross_entropy = tf.reduce_sum(sigmoid_cross_entropy_per_example, reduction_indices=1, name='cross_entropy')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
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


def train(total_loss, global_step):
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






