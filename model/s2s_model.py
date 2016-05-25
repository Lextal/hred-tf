import tensorflow as tf
from tensorflow.python.ops.rnn import *
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops import variable_scope, seq2seq
import numpy as np
import numpy.random as rnd

from functools import reduce


class HierarchicalSeq2SeqModel:
    def __init__(self, vocab_size, batch_size, topology, cells_size,
                 learning_rate, lr_decay_rate, max_gradient_norm,
                 embed=False, forward_only=False):
        self.emb_size = vocab_size
        self.batch_size = batch_size
        self.topology = topology
        self.n_layers = len(topology)
        self.cells_size = cells_size

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * lr_decay_rate)
        self.global_step = tf.Variable(0, trainable=False)
        seq_len = reduce(lambda x, y: x * y, self.topology)
        self.enc_inputs = [tf.placeholder(tf.float32, [batch_size, self.emb_size],
                                          name='Encoder_Input_{}'.format(q)) for q in range(seq_len)]
        self.dec_inputs = []

        self.enc_cells = []
        self.dec_cells = []

        self.enc_scopes = []
        self.dec_scopes = []
        self.dec_data = []

        for i in range(self.n_layers):
            enc_cell = BasicLSTMCell(((2 * self.cells_size) ** i) * self.emb_size)
            if self.cells_size > 1:
                cell = MultiRNNCell([enc_cell] * self.cells_size)
                self.enc_cells.append(cell)
            else:
                self.enc_cells.append(enc_cell)

            self.enc_scopes.append('encoder_{}'.format(i))

            dec_cell = BasicLSTMCell(((2 * self.cells_size) ** i) * self.emb_size)
            if self.cells_size > 1:
                cell = MultiRNNCell([dec_cell] * self.cells_size)
                self.dec_cells.append(cell)
            else:
                self.dec_cells.append(dec_cell)

            j = self.n_layers - i - 1

            dec_inp = [tf.placeholder(tf.float32, [batch_size, (2 ** j) * self.cells_size * self.emb_size])
                       for _ in range(self.topology[j])]
            dec_data = [np.zeros((batch_size, (2 ** j) * self.cells_size * self.emb_size))
                        for _ in range(self.topology[j])]
            self.dec_inputs.append(dec_inp)
            self.dec_data.append(dec_data)
            self.dec_scopes.append('decoder_{}'.format(i))

        self.dec_inputs = self.dec_inputs[::-1]
        self.dec_data = self.dec_data[::-1]
        self.dec_cells = self.dec_cells[::-1]

        if embed:
            self.enc_cells[0] = EmbeddingWrapper(self.enc_cells[0], self.emb_size, self.emb_size)
            self.enc_inputs = [tf.placeholder(tf.int32, [None],
                                              name='Encoder_Input_{}'.format(q)) for q in range(seq_len)]
            self.targets = [tf.placeholder(tf.int32, [None],
                                           name='Target_{}'.format(q)) for q in range(seq_len)]
            self.weights = [tf.placeholder(tf.float32, 1, name='Weights_{}'.format(q)) for q in range(seq_len)]

        self.default_weights = [np.asarray([1.]) for _ in range(seq_len)]
        self.encoder = self.hierarchical_encoder()
        self.seq2seq = self.hierarchical_decoder(self.encoder)
        self.losses = seq2seq.sequence_loss(self.seq2seq, self.targets, self.weights)

        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.AdagradOptimizer(self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.gradient_norm = norm
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    @staticmethod
    def encoder(cell, inputs, n_steps, batch_size=1, dtype=tf.float32, scope=None):
        states = []
        with variable_scope.variable_scope(scope):
            init_state = cell.zero_state(batch_size, dtype)
            for i in range(0, len(inputs), n_steps):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                _, state = rnn(cell, inputs[i: i + n_steps], init_state, dtype)
                states.append(state)
        return states

    def hierarchical_encoder(self):
        with variable_scope.variable_scope('encoder'):
            states = self.enc_inputs
            for i in range(self.n_layers):
                states = self.encoder(self.enc_cells[i],
                                      states,
                                      self.topology[i],
                                      self.batch_size,
                                      tf.float32,
                                      self.enc_scopes[i])
        return states

    @staticmethod
    def decoder(cell, dec_outputs, states, scope):
        outputs = []
        with variable_scope.variable_scope(scope):
            for i in range(len(states)):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                outs, _ = seq2seq.rnn_decoder(dec_outputs, states[i], cell)
                outputs.extend(outs)
        return outputs

    def hierarchical_decoder(self, embedding):
        with variable_scope.variable_scope('decoder'):
            dec = embedding
            for i in range(self.n_layers):
                dec = self.decoder(self.dec_cells[i], self.dec_inputs[i], dec, self.dec_scopes[i])
        return dec

    def step(self, session, inputs, outputs, forward_only=False):
        """

        :param session: Current session to run a step in
        :param inputs: List of buckets, each element within is int32, structure conforming to enc_inputs
        :param outputs: List of target outputs, structure is the same as inputs
        :param forward_only: Boolean, indicating whether to perform backward pass
        :return:
        """
        if len(inputs) != len(self.enc_inputs):
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d.".format(len(self.enc_inputs), len(inputs)))
        if len(outputs) != len(self.targets):
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d.".format(len(self.targets), len(outputs)))

        feed = {}
        # adding input sequence for encoder
        for ph, tensor in zip(self.enc_inputs, inputs):
            feed[ph] = tensor
        # output sequences for every decoder (filled with zeros)
        for i in range(self.n_layers):
            for j, tensor in enumerate(self.dec_inputs[i]):
                feed[tensor] = self.dec_data[i][j]
            # target sequence
            for ph, tensor in zip(self.targets, outputs):
                feed[ph] = tensor
            # weights
            for ph, tensor in zip(self.weights, self.default_weights):
                feed[ph] = tensor

        output_feed = []
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.losses]  # Loss for this batch.
        else:
            output_feed.append(self.losses)  # only loss
        output_feed.extend(self.seq2seq)

        results = session.run(output_feed, feed)
        return results[1], results[2:]

    def get_batch(self, data, size=None):
        if not size:
            size = self.batch_size

        seq_size = len(self.enc_inputs)
        ind = rnd.randint(0, len(data[0]), size).tolist()
        input_sample = [data[0][i] for i in ind]
        output_sample = [data[1][i] for i in ind]
        inputs = [[] for _ in range(seq_size)]
        targets = [[] for _ in range(seq_size)]

        for i in range(seq_size):
            for j in range(size):
                inputs[i].append(input_sample[j][i])
                targets[i].append(output_sample[j][i])

        inputs = map(lambda x: np.asarray(x, dtype='int32'), inputs)
        targets = map(lambda x: np.asarray(x, dtype='int32'), targets)
        return list(inputs), list(targets)
