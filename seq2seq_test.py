import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = 20
batch_size = 100


def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)
            ]


batches = random_sequences(length_from=3, length_to=10,
                           vocab_lower=2, vocab_upper=10,
                           batch_size=batch_size)


def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths


train_graph = tf.Graph()
with train_graph.as_default():
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
    )
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="plain_decoder",
    )

    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    decoder_prediction = tf.argmax(decoder_logits, 2)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

loss_track = []
epochs = 3001

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch = next(batches)
        encoder_inputs_, _ = make_batch(batch)
        decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
        decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
        feed_dict = {encoder_inputs: encoder_inputs_, decoder_inputs: decoder_inputs_,
                     decoder_targets: decoder_targets_,
                     }
        _, l = sess.run([train_op, loss], feed_dict)
        loss_track.append(l)
        if epoch == 0 or epoch % 1000 == 0:
            print('loss: {}'.format(sess.run(loss, feed_dict)))
            predict_ = sess.run(decoder_prediction, feed_dict)
            for i, (inp, pred) in enumerate(zip(feed_dict[encoder_inputs].T, predict_.T)):
                print('input > {}'.format(inp))
                print('predicted > {}'.format(pred))
                if i >= 20:
                    break

plt.plot(loss_track)
plt.show()
