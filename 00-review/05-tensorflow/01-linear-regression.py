import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

def get_data():
    x = np.linspace(-1, 1, 100)
    y = 2 * x + 0.3 * np.random.randn(*x.shape)

    return x, y


def model(graph):
    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None,), name='x')
        y = tf.placeholder(dtype=tf.float32, shape=(None,), name='y')

        with tf.variable_scope('LinReg'):
            w = tf.Variable(np.random.rand(), dtype=tf.float32, name="wieght")
            b = tf.Variable(np.random.rand(), dtype=tf.float32, name="bias")

            pred = tf.multiply(w, x) + b
            loss = tf.reduce_mean(0.5 * tf.square(y - pred))
    return x, y, pred, loss


def run():
    learning_rate = 0.1
    epochs = 100
    x_train, y_train = get_data()

    g1 = tf.Graph()
    x, y, pred, loss = model(g1)

    with g1.as_default():
        init = tf.global_variables_initializer()
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate
        ).minimize(loss)

        with tf.Session() as sess:
            sess.run(init)
            feed_dict = {x: x_train, y: y_train}

            for i in range(epochs):
                loss_val, _ = sess.run([loss, optimizer], feed_dict)
                print("epoch " , i, " : ",  loss_val)

            y_pred = sess.run(pred, {x: x_train})

    plt.figure()
    plt.plot(x_train, y_train, 'ro', label='original data')
    plt.plot(x_train, y_pred)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()
