"""An Example of a DNNClassifier for the Iris dataset."""

import argparse
import tensorflow as tf
import logging
import sys

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data, 通过我们创建的数据集
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # 定义特征列
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 units each
        hidden_units=[10, 10],
        # The model must choose between 3 classes
        n_classes=3
    )

    print('train_steps', args.train_steps)

    # Train the model
    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps
    )


    # Evaluate the Model
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size))

    print('\n Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x, labels=None,
                                                 batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id], 100 * probability,
                              expec))


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
