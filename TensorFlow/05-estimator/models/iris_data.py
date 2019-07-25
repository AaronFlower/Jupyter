import pandas as pd
import tensorflow as tf
from tensorflow import keras

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth',
                    'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Species'):
    """
    Returns the iris dataset as (train_x, train_y), (test_x, test_y)
    """
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, name=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, name=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """ An input function for training """
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples
    # 先对数据进行下 shuffle,
    # 然后对数据进行重复，repeat() 不传入重复次数时，默认是无限次重复
    # 然后将整个数据集进行分布大小为 batch_size 的 batch
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    """ An input function for evaluation and prediction """
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset
    dataset = tf.data.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset
