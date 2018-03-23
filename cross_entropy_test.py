import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def test_cross_entropy():
    x_data = np.linspace(0, 0.5, 200)[:, None]
    noise_data = np.random.uniform(-0.02, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise_data

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    weight_layer1 = tf.Variable(tf.random_normal([1, 100]))
    output_layer1 = tf.nn.sigmoid(tf.matmul(x, weight_layer1))

    weight_layer2 = tf.Variable(tf.random_normal([100, 1]))
    logits = tf.matmul(output_layer1, weight_layer2)
    predicts = tf.nn.sigmoid(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(1, 10000):
            session.run(train, feed_dict={x: x_data, y: y_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.scatter(x_data, session.run(predicts, feed_dict={x: x_data, y: y_data}), c="r")
        plt.show()


if __name__ == "__main__":
    test_cross_entropy()
