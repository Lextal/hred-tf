import tensorflow as tf
from tensorflow.python.ops.rnn import *
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops import variable_scope, seq2seq
import numpy as np
import numpy.random as rnd


class HierarchicalSeq2SeqModel:
    def __init__(self, vocab_size, batch_size, topology, cell_sizes,
                 learning_rate, lr_decay_rate, max_gradient_norm,
                 cell_type=BasicLSTMCell, embed=False, forward_only=False):
        self.emb_size = vocab_size
        self.batch_size = batch_size
        self.seq_sizes = topology
        self.n_layers = len(topology)
        self.cell_sizes = cell_sizes

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * lr_decay_rate)
        self.global_step = tf.Variable(0, trainable=False)
        self.seq_len = 1
        for seq_size in self.seq_sizes:
            self.seq_len *= seq_size
        self.enc_inputs = [tf.placeholder(tf.float32, [batch_size, self.emb_size],
                                          name='Encoder_Input_{}'.format(q)) for q in range(self.seq_len)]
        self.dec_inputs = []
        self.enc_cells = []
        self.dec_cells = []
        self.enc_scopes = []
        self.dec_scopes = []
        self.dec_data = []

        self.cell_type = cell_type

        # topology = [..., (layer_size, state_dim), ...]

        def build_layer(layer_size, input_size):
            enc_cell = self.cell_type(input_size)
            if layer_size > 1:
                enc_cell = [enc_cell]
                for _ in range(1, layer_size):
                    enc_cell.append(self.cell_type(input_size, enc_cell[-1].output_size))
                enc_cell = MultiRNNCell(enc_cell)
            return enc_cell

        def build_inputs(seq_len, input_size):
            return [tf.placeholder(tf.float32, [self.batch_size, input_size]) for _ in range(seq_len)]

        for i in range(0, self.n_layers):
            size = self.enc_cells[i - 1].state_size if i > 0 else self.emb_size
            cell = build_layer(self.cell_sizes[i], size)
            self.enc_cells.append(cell)
            self.enc_scopes.append('encoder_{}'.format(i))
            dec_input = build_inputs(self.seq_sizes[i], size)
            self.dec_cells.append(cell)
            self.dec_inputs.append(dec_input)
            self.dec_data.append([np.zeros((batch_size, self.dec_cells[i].input_size))
                                  for _ in range(self.seq_sizes[i])])
            self.dec_scopes.append('decoder_{}'.format(i))

        self.dec_inputs = self.dec_inputs[::-1]
        self.dec_data = self.dec_data[::-1]
        self.dec_cells = self.dec_cells[::-1]

        if embed:
            self.enc_cells[0] = EmbeddingWrapper(self.enc_cells[0], self.emb_size, self.emb_size)
            self.enc_inputs = [tf.placeholder(tf.int32, [None],
                                              name='Encoder_Input_{}'.format(q)) for q in range(self.seq_len)]
            self.targets = [tf.placeholder(tf.int32, [None],
                                           name='Target_{}'.format(q)) for q in range(self.seq_len)]
            self.weights = [tf.placeholder(tf.float32, [None],
                                           name='Weights_{}'.format(q)) for q in range(self.seq_len)]

        self.encoder = self.hierarchical_encoder()
        self.logits = self.hierarchical_decoder(self.encoder)
        self.seq2seq = [tf.arg_max(x, 1) for x in self.logits]
        self.losses = seq2seq.sequence_loss(self.logits, self.targets, self.weights)

        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
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
                n_steps = self.seq_sizes[i]
                states = self.encoder(self.enc_cells[i],
                                      states,
                                      n_steps,
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

    def step(self, session, inputs, outputs, weights, forward_only=False):
        """
        :param session: Current session to run a step in
        :param inputs: List of buckets, each element within is int32, structure conforming to enc_inputs
        :param outputs: List of target outputs, structure is the same as inputs
        :param weights:
        :param forward_only: Boolean, indicating whether to perform backward pass
        :return:
        """
        if len(inputs) != len(self.enc_inputs):
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " {} != {}.".format(len(self.enc_inputs), len(inputs)))
        if len(outputs) != len(self.targets):
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " {} != {}.".format(len(self.targets), len(outputs)))

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
            for ph, tensor in zip(self.weights, weights):
                feed[ph] = tensor

        output_feed = []
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.losses]  # Loss for this batch.
        output_feed.extend(self.seq2seq)

        results = session.run(output_feed, feed)
        if not forward_only:
            return results[1], results[2:]
        else:
            return results

    def get_batch(self, data, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size

        seq_size = len(self.enc_inputs)
        ind = rnd.randint(0, len(data[0]), batch_size).tolist()
        input_sample = [data[0][i] for i in ind]
        output_sample = [data[1][i] for i in ind]
        inputs = [[] for _ in range(seq_size)]
        targets = [[] for _ in range(seq_size)]

        for i in range(seq_size):
            for j in range(batch_size):
                inputs[i].append(input_sample[j][i])
                targets[i].append(output_sample[j][i])

        weights = [np.sign(x).astype(float) for x in targets]
        return inputs, targets, weights, ind
