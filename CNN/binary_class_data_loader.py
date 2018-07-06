import numpy as np

from tensorflow.contrib import learn

class BinaryClassDataLoader(object):
    """
    Load binary classification data from two files (positive and negative) and
    split data into train and dev.
    """
    def __init__(self, flags, data_processor, clean_data=None, classes=None):
        self.__flags = flags
        self.__data_processor = data_processor
        self.__clean_data = clean_data
        self.__classes = classes

    def define_flags(self):
        self.__flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
        self.__flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
        self.__flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

    def prepare_data(self):
        self.__resolve_params()

        x_text, y = self.load_data_and_labels()

        # Build vocabulary
        self.vocab_processor = self.__data_processor.vocab_processor(x_text)
        x = np.array(list(self.vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(self.__dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        return [x_train, y_train, x_dev, y_dev]

    def restore_vocab_processor(self, vocab_path):
        self.vocab_processor = self.__data_processor.restore_vocab_processor(vocab_path)
        return self.vocab_processor

    def class_labels(self, class_indexes):
        if self.__classes is None:
            result = class_indexes
        else:
            result = [ self.__classes[idx] for idx in class_indexes ]
        return result

    def load_data_and_labels(self):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        self.__resolve_params()

        # Load data from files
        positive_examples = list(open(self.__positive_data_file, "r").readlines())
        negative_examples = list(open(self.__negative_data_file, "r").readlines())
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.__data_processor.clean_data(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def __resolve_params(self):
        self.__dev_sample_percentage = self.__flags.FLAGS.dev_sample_percentage
        self.__positive_data_file = self.__flags.FLAGS.positive_data_file
        self.__negative_data_file = self.__flags.FLAGS.negative_data_file
