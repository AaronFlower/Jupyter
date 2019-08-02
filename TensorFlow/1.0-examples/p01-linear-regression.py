import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_eager_execution()


def get_data():
    x = np.linspace(-1, 1, 100)
    y = 2 * x + 0.3 * np.random.randn(*x.shape)
    return x, y


def model():
    x = tf.placeholder(tf.float32, shape=(None,), name='x')
    y = tf.placeholder(tf.float32, shape=(None,), name='y')

    with tf.variable_scope('lreg'):
        w = tf.Variable(np.random.normal(), name='weight')
        b = tf.Variable(np.random.normal(), name='bias')

        y_pred = tf.multiply(w, x) + b
        loss = tf.reduce_mean(tf.square(y_pred - y))

    return x, y, y_pred, loss


def run():
    x_train, y_train = get_data()
    epochs = 100
    learning_rate = 0.1

    x, y, y_pred, loss = model()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        feed_dict = {x: x_train, y: y_train}

        for epoch in range(epochs):
            loss_val, _ = sess.run([loss, optimizer], feed_dict)
            if (epoch % 20 == 0):
                print('loss = ', loss_val)

        y_pred_batch = sess.run(y_pred, {x: x_train})

    print('Last loss:', loss_val)

    plt.figure(1)
    plt.plot(x_train, y_train, 'ro', label='Original Data')
    plt.plot(x_train, y_pred_batch)
    plt.legend()
    plt.savefig('./linear-regression.png')


if __name__ == "__main__":
    run()
