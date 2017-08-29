import tensorflow as tf
import logging
import time
import numpy as np


class RNNModel(object):
    def __init__(self, is_test, hidden_size, rnn_layers, batch_size, seq_length, vocab_size, embedding_size,
                 learning_rate, max_grad_norm):
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = tf.constant(learning_rate)

        if is_test == 'true':
            self.batch_size = 1
            self.seq_length = 1
        cell_fn = tf.contrib.rnn.BasicRNNCell
        cell = cell_fn(hidden_size)
        cells = [cell]
        for i in range(rnn_layers - 1):
            higher_layer_cell = cell_fn(hidden_size)
            cells.append(higher_layer_cell)

        multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

        self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)

        self.initial_state = create_tuple_placeholders_with_default(self.zero_state, shape=multi_cell.state_size)
        self.input_data = tf.placeholder(tf.int64, [self.batch_size, self.seq_length], name='inputs')
        self.targets = tf.placeholder(tf.int64, [self.batch_size, self.seq_length], name='targets')

        self.embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        sliced_inputs = [tf.squeeze(input_, [1]) for input_ in
                         tf.split(axis=1, num_or_size_splits=self.seq_length, value=inputs)]
        outputs, final_state = tf.contrib.rnn.static_rnn(multi_cell, sliced_inputs, initial_state=self.initial_state)
        self.final_state = final_state

        flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
        flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])

        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
        # self.probs = tf.nn.softmax(self.logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=flat_targets)
        mean_loss = tf.reduce_mean(loss)

        count = tf.Variable(1.0, name='count')
        sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
        update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss + mean_loss), count.assign(count + 1),
                                       name='update_loss_monitor')
        with tf.control_dependencies([update_loss_monitor]):
            self.average_loss = sum_mean_loss / count

        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(0.0))

        if is_test == 'false':
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(mean_loss, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def train(self, session, train_size, train_batches):
        epoch_size = train_size // (self.batch_size * self.seq_length)
        if train_size % (self.batch_size * self.seq_length) != 0:
            epoch_size += 1
        state = session.run(self.zero_state)
        start_time = time.time()
        for step in range(epoch_size):
            data = train_batches.next()
            inputs = np.array(data[:-1]).transpose()
            targets = np.array(data[1:]).transpose()
            ops = [self.average_loss, self.final_state, self.train_op, self.global_step, self.learning_rate]
            feed_dict = {self.input_data: inputs, self.targets: targets,
                         self.initial_state: state}
            average_loss, state, __, global_step, lr = session.run(ops, feed_dict)
        logging.info("average loss: %.3f, speed: %.0f chars per sec",
                     average_loss, (step + 1) * self.batch_size * self.seq_length /
                     (time.time() - start_time))

    def predict(self, session, start_text, length, vocab_index_dict, index_vocab_dict):
        state = session.run(self.zero_state)
        seq = list(start_text)
        for char in start_text[:-1]:
            x = np.array([[vocab_index_dict[char]]])
            state = session.run(self.final_state, {self.input_data: x, self.initial_state: state})
        x = np.array([[vocab_index_dict[start_text[-1]]]])
        for i in range(length):
            state, logits = session.run([self.final_state, self.logits],
                                        {self.input_data: x, self.initial_state: state})
            unnormalized_probs = np.exp(logits - np.max(logits))
            probs = unnormalized_probs / np.sum(unnormalized_probs)
            sample = np.argmax(probs[0])
            seq.append(index_vocab_dict[sample])
            x = np.array([[sample]])
        print(''.join(seq))


def create_tuple_placeholders_with_default(inputs, shape):
    if isinstance(shape, int):
        result = tf.placeholder_with_default(
            inputs, list((None,)) + [shape])
    else:
        subplaceholders = [create_tuple_placeholders_with_default(
            subinputs, subshape)
                           for subinputs, subshape in zip(inputs, shape)]
        t = type(shape)
        if t == tuple:
            result = t(subplaceholders)
        else:
            result = t(*subplaceholders)
    return result
