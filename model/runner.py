import math
import os
from time import time

import numpy as np
import tensorflow as tf

from s2s_model import HierarchicalSeq2SeqModel

data_dir = '../../../data/test/'
train_dir = data_dir

DEFAULT_VOCAB = [''] + [chr(ord('a') + i) for i in range(26)] + [' ']

"""
    Topology describes the length of a sequence on a certain level of hierarchy.
    For example, topology = [10, 5] represents a two-level hierarchy with
        maximal length of sequence 10 * 5 = 50, where the first level generates representations of
        10-symbol subsequences and the second - a final representation from 5-symbol one.
    Amount of symbols on every data point must be equal to the product of topology layers' sizes.
"""

topology = [8, 10]
seq_len = 1  # computing fixed length of a sequence
for q in topology:
    seq_len *= q

tf.app.flags.DEFINE_string('train_dir', train_dir, "Model directory")
tf.app.flags.DEFINE_integer('vocab_size', len(DEFAULT_VOCAB), "The size of vocabulary")
tf.app.flags.DEFINE_integer('batch_size', 50, "The size of batch for every step")
tf.app.flags.DEFINE_integer("num_layers", 5, "Number of LSTM cells in each layer")
tf.app.flags.DEFINE_float("learning_rate", 0.8, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("seq_len", seq_len, "Fixed length of input and output sequence")

FLAGS = tf.app.flags.FLAGS


def read_data(source_path, target_path):
    inputs = []
    outputs = []
    with tf.gfile.GFile(source_path) as source:
        with tf.gfile.GFile(target_path) as target:
            s, t = source.readline(), target.readline()
            counter = 0
            while s and t:
                counter += 1
                if counter % 10000 == 0:
                    print('Reading data line {}'.format(counter))
                source_ids = [int(x) for x in s.split()]
                target_ids = [int(x) for x in t.split()]
                inputs.append(source_ids)
                outputs.append(target_ids)
                s, t = source.readline(), target.readline()
    return inputs, outputs


def create_model(session, forward_only):
    model = HierarchicalSeq2SeqModel(FLAGS.vocab_size, FLAGS.batch_size, topology,
                                     FLAGS.num_layers, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                                     FLAGS.max_gradient_norm, embed=True, forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train(source_path, target_path):
    with tf.Session() as sess:
        model = create_model(sess, False)

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        dataset = read_data(source_path, target_path)

        while True:
            start_time = time()
            data, targets, weights, ind = model.get_batch(dataset)
            step_loss, outputs = model.step(sess, data, targets, weights)
            step_time += (time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "hred.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0


def decode(outputs, vocab=DEFAULT_VOCAB):
    """
    Super simple decoder for two-level hierarchy (chars -> words)
    """
    seqs = np.asarray(outputs).T
    results = []
    for seq in seqs:
        results.append(''.join([vocab[x] for x in seq]))
    return results


'''
    Both source.txt and target.txt are files of equal length with sequences of integers
    One sequence - one line
    It's assumed that all sequences conform to topology
'''
if __name__ == '__main__':
    source_path = FLAGS.train_dir + 'source.txt'
    target_path = FLAGS.train_dir + 'target.txt'
    train(source_path, target_path)
    # sample_decoding(source_path, target_path)
