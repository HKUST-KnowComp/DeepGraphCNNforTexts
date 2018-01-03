
## data
ORI_DATA_NAME = 'graphs'
ORI_TRAIN_DATA_NAME = 'train_graphs'
ORI_TEST_DATA_NAME = 'test_graphs'
ORI_DATA_VEC_NAME = 'index2vec'
ORI_DATA_OPTION_NAME = 'option'

TRAIN_DATA_NAME = 'data.train'
TEST_DATA_NAME = 'data.test'
DATA_OPTION_NAME = 'data.option'

DATA_LABELS_REMAP_NAME = 'remap'

## LSHTC Hierarchy training


HIER_used = True
HIER_test_used = True
rootstr = '_1_2322682_' # ????
HIER_ROOT_CODE = [2322682] # ????
HIER_DIR_NAME = 'hier'
HIER_labels_remap_file = 'hier'+rootstr+'remap'
HIER_train_graphs_index_file = 'hier'+rootstr+'train_graphs_index'
HIER_train_labels_file = 'hier'+rootstr+'train_labels'
HIER_train_data_file = 'hier'+rootstr+'train_data'  # ??
HIER_test_graphs_index_file = 'hier'+rootstr+'test_graphs_index'
HIER_test_labels_file = 'hier'+rootstr+'test_labels'
HIER_test_data_file = 'hier'+rootstr+'test_data'  # ??

HIER_eval_result_leaf_file = 'hier_eval_result'+rootstr+'leaf'
HIER_eval_result_leaf_exp_file = 'hier_eval_result'+rootstr+'leaf_exp'
HIER_eval_result_root_file = 'hier_eval_result'+rootstr+'root'

if HIER_used:
    TRAIN_DATA_NAME = HIER_train_data_file
    if HIER_test_used:
        TEST_DATA_NAME = HIER_test_data_file




# lr_decay_value = [0.1,0.01,0.001,0.0005,0.0001] # single-label wiki_cn
# lr_decay_ecophs = [2,150,750,1250,1500]   # single-label wiki_cn
# lr_decay_value = [0.1,0.01,0.001,0.01,0.001,0.0001]
lr_decay_value = [0.01,0.001,0.0001,0.01,0.001,0.0001,0.00001]
# lr_decay_ecophs = [10,400,1500,1800,2000]   # multi-label, RCV
lr_decay_ecophs = [1,300,600,601,1000,1400,1500]   # multi-label, RCV

# multi-label, RCV: INITIAL_LEARNING_RATE = 0.001, decay_epochs = 600



## Basic parameters.
TRAIN_DATA_DIR = '../graphCNN_data'  # Path to the train data directory.
EVAL_DATA_DIR = '../graphCNN_data'  # Path to the test data directory.
DATA_PATH = './data'   # Path to data directory

USE_FP16 = False  # Train the model using fp16.

# summaryWriter
SUMMARYWRITER = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'



## model parameters
NUM_EPOCHS_PER_DECAY = 1000 #350     # Epochs after which learning rate decays.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.
LEARNING_RATE_DECAY_RATE = 0.1  # Learning rate decay rate.

MOMENTUM = 0.9 # Momentum of SGD

DROPOUT_FRACTION = 0.5 # Add a dropout during training.

MOVING_AVERAGE_DECAY = 0.999 # The decay to use for the moving average.

WEIGHT_DECAY = 0.0005     # 0.00005  # 0.0005 # l2 regularization weight decay

VARIABLE_DEPENDENCY = 0.00005 # 0.0005 # the Variable's dependency constraint


## train parameters
NUM_GPUS = 4 # How many GPUs to use

CKPT_PERIOD = 5000


## eval parameters
EVALUTION_THRESHOLD_FOR_MULTI_LABEL = 0.5 # the evalution threshold for multi-label classification
