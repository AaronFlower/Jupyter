import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def load_dataset(epochs=1, batch_size=100):
    """
    根据  epochs, batch_size 来构建训练和测试 dataset
    """
    mnist = tf.keras.datasets.mnist
    train, test = mnist.load_data()
    n_train_samples = train[0].shape[0]
    n_test_samples = test[0].shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices(train)
    train_dataset = train_dataset.map(
        lambda x, y: (tf.reshape(x, shape=(784,)), tf.one_hot(y, 10))
    )
    # break a line of chained methods
    train_dataset = (
        train_dataset.shuffle(n_train_samples)
        .repeat(epochs)
        .batch(batch_size)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(test)
    test_dataset = test_dataset.map(
        lambda x, y: (tf.reshape(x, shape=(784,)), tf.one_hot(y, 10))
    ).batch(batch_size * 10)

    return (train_dataset, n_train_samples), (test_dataset, n_test_samples)


def model(graph):
    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

        with tf.variable_scope('lr'):
            # w = tf.Variable(np.random.randn(784, 10), dtype=tf.float32,
            w = tf.Variable(tf.zeros((784, 10)), dtype=tf.float32,
                            name='weights')
            b = tf.Variable(tf.zeros(10), dtype=tf.float32)
            pred = tf.nn.softmax(tf.matmul(x, w) + b)
            cross_entropy = -tf.reduce_sum(
                y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)),
                reduction_indices=[1],
            )
            loss = tf.reduce_mean(cross_entropy)

    return x, y, pred, loss


def run():
    learning_rate = 0.01
    epochs = 20
    batch_size = 100

    g1 = tf.Graph()
    with g1.as_default():
        x, y, pred, loss = model(g1)
        train, test = load_dataset(
            epochs=epochs,
            batch_size=batch_size
        )
        train_dataset, n_train_samples = train
        train_iterator = tf.data.make_one_shot_iterator(train_dataset)
        next_train = train_iterator.get_next()

        test_dataset, n_test_samples = test
        next_test = tf.data.make_one_shot_iterator(test_dataset).get_next()

        # optimizer = (
        #      tf.train.GradientDescentOptimizer(learning_rate)
        #      .minimize(loss)
        #  )
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Train
            for i in range(epochs):
                n_batches = int(n_train_samples / batch_size)
                avg_loss = 0
                for k in range(n_batches):
                    x_train, y_train = sess.run(next_train)
                    feed_dict = {x: x_train, y: y_train}
                    _, loss_val = sess.run([optimizer, loss], feed_dict)
                    avg_loss += loss_val / n_batches

                print('Epoch %3d : %3.3f' % (i, avg_loss))

            # Test
            acc = 0
            try:
                while True:
                    x_test, y_test = sess.run(next_test)
                    y_pred = sess.run(pred, {x: x_test})
                    equal_check = tf.equal(
                        tf.argmax(y_test, axis=1),
                        tf.argmax(y_pred, axis=1)
                    )
                    acc += sess.run(
                        tf.reduce_sum(tf.cast(equal_check, dtype=tf.int32))
                    )
                    print('acc: %d' % acc)
            except tf.errors.OutOfRangeError:
                print('test finished')

            print('Correct predictions: ', acc)
            print('Accuracy: %2.2f' % ((acc/n_test_samples) * 100))
            # print('Accuracy: %2.2f' % (acc/500))


if __name__ == "__main__":
    #  TODO: 训练和测试分开，怎么将参数保存下来那？  #
    run()
