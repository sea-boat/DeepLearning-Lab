import codecs
import numpy as np
import logging
import time
import tensorflow as tf
import sys
import argparse
import json
from six import iteritems
from rnn_model import RNNModel

data_file = "data/rnn-train-data.txt"
rnn_layers = 2
embedding_size = 128
hidden_size = 128
input_dropout = 0.2
learning_rate = 0.01
max_grad_norm = 5
num_epochs = 501
batch_size = 20
seq_length = 10
rnn_model = "D:\kuaipan\workspace\DeepLearning-Lab\model/rnn\model.ckpt"
restore_path = 'D:\kuaipan\workspace\DeepLearning-Lab\model/rnn/'


def main():
    args = parse_args()
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO,
                        datefmt='%I:%M:%S')
    with codecs.open(data_file, 'r') as f:
        text = f.read()
    train_size = len(text)
    train_text = text
    if args.test == 'false':
        vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(text)
        save_vocab(vocab_index_dict, 'vocab.json')
    else:
        vocab_index_dict, index_vocab_dict, vocab_size = load_vocab('vocab.json')

    train_batches = BatchGenerator(train_text, batch_size, seq_length, vocab_size, vocab_index_dict)
    graph = tf.Graph()
    with graph.as_default():
        model = RNNModel(args.test, hidden_size, rnn_layers, batch_size, seq_length, vocab_size, embedding_size,
                         learning_rate, max_grad_norm)

    with tf.Session(graph=graph) as session:
        model_saver = tf.train.Saver()
        if args.test == 'false':
            tf.global_variables_initializer().run()
            for i in range(num_epochs):
                model.train(session, train_size, train_batches)
                if i % 100 == 0:
                    logging.info("saving model")
                    model_saver.save(session, rnn_model, global_step=model.global_step)
        else:
            module_file = tf.train.latest_checkpoint(restore_path)
            model_saver.restore(session, module_file)
            start_text = 'your'
            length = 20
            print(model.predict(session, start_text, length, vocab_index_dict, index_vocab_dict))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help=help, default='false')
    args = parser.parse_args()
    return args


def load_vocab(vocab_file):
    with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in iteritems(vocab_index_dict):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def save_vocab(vocab_index_dict, vocab_file):
    with codecs.open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_index_dict, f, indent=2, sort_keys=True)


def create_vocab(text):
    unique_chars = list(set(text))
    print(unique_chars)
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


class BatchGenerator(object):
    def __init__(self, text, batch_size, seq_length, vocab_size, vocab_index_dict):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.vocab_index_dict = vocab_index_dict

        segment = self._text_size // batch_size

        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        batch = np.zeros(shape=(self._batch_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b] = self.vocab_index_dict[self._text[self._cursor[b]]]
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        batches = [self._last_batch]
        for step in range(self.seq_length):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


if __name__ == '__main__':
    main()
