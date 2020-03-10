# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The expermientdata required for this example is in the expermientdata/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/expermientdata/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

# from tensorflow.models.rnn.ptb import reader
import myreader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("checkpoint_dir", "ckpt", "checkpoint_dir")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("train", False,
                  "should we train or test")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        # batch_size = 1
        self.batch_size = batch_size = config.batch_size
        # num_steps = 1
        self.num_steps = num_steps = config.num_steps

        # size = hidden_size = 200 = um_units = num_neurons
        size = config.hidden_size

        # vocab_size = 10000
        vocab_size = config.vocab_size

        # [batch_size, num_steps] = [1, 1]
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])  

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        # size = hidden_size = um_units = num_neurons = 200
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # not for test
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)   

        # config.num_layers = 2
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True) 

        self._initial_state = cell.zero_state(batch_size, data_type()) 

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())  
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)  

            # shape = (10000, 200)
            self._embedding = embedding

            # shape = (1, 1, 200) (batch_size, num_steps, num_units)
            self._inputs = inputs

        # not for test
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob) 
        self._inputs_dropout = inputs

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn

        squeeze_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, num_steps, 1)] 

        self._squeeze_inputs = squeeze_inputs

        outputs, state = tf.nn.static_rnn(cell, squeeze_inputs, initial_state=self._initial_state)
        self._above_neurons = []
        self._below_neurons = []
        # self.__nc, self._above_neurons, self._below_neurons = self.get_neuron_coverage(state)
        self._outputs = outputs
        self._nc_tracker = []

        outputs_all = []
        states_all = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state) 
                outputs_all.append(cell_output)
                states_all.append(state)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        # output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        self._outputs_all = outputs_all
        self._state = state
        self._states_all = states_all

        self._output = output

        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b

        self.sample = tf.multinomial(logits, 1)  # this is the sampling operation

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])

        self._loss = loss
        self._reduce_sum_loss = tf.reduce_sum(loss)

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        # RANI
        self.logits = logits

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        for v in tvars:
            print(v)

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)

        # self._grads = grads

        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def outputs_all(self):
        return self._outputs_all

    @property
    def state(self):
        return self._state

    @property
    def states_all(self):
        return self._states_all

    @property
    def inputs_dropout(self):
        return self._inputs_dropout

    @property
    def nc_tracker(self):
        return self._nc_tracker

    @property
    def above_neurons(self):
        return self._above_neurons

    @property
    def below_neurons(self):
        return self._below_neurons

    @property
    def nc(self):
        return self._nc

    @property
    def grads(self):
        return self._grads

    @property
    def loss(self):
        return self._loss

    @property
    def reduce_sum_loss(self):
        return self._reduce_sum_loss

    @property
    def embedding(self):
        return self._embedding

    @property
    def inputs(self):
        return self._inputs

    @property
    def squeeze_inputs(self):
        return self._squeeze_inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def output(self):
        return self._output

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1       
    learning_rate = 1.0
    max_grad_norm = 5   
    num_layers = 2    
    num_steps =20  
    hidden_size = 200   
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5      
    batch_size = 20    
    vocab_size = 10000   


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, inverseDictionary, data, eval_op, verbose=False):
    """Runs the model on the given expermientdata."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)


    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):

        grads = tf.gradients(model.outputs, model.inputs)

        fetches = [model.cost, model.final_state, model.logits, eval_op, model.embedding, model.inputs, grads, model.initial_state, model.outputs_all, model.outputs, model.state, model.states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        cost, state, logits, _, val_1, val_2, val_3, initial_state, outputs_try, outputs, state, state_try = session.run(fetches, feed_dict)
        print("cost: ", cost)

        costs += cost
        iters += model.num_steps
        # Rani: show the actual prediction
        # decodedWordId = int(np.argmax(logits))

        # if int(decodedWordId) not in inverseDictionary or int(y) not in inverseDictionary:
        #     print("not in inverseDictionary")
        # else:
        #     print(" ".join([inverseDictionary[int(x1)] for x1 in np.nditer(x)]) + " got:" + inverseDictionary[\
        #         decodedWordId] + " expected:" + inverseDictionary[int(y)])

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


# Rani: save the session
def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB expermientdata directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _, word_to_id = raw_data
    # Rani: added inverseDictionary

    inverseDictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    config = get_config()
    eval_config = get_config()

    eval_config.batch_size = 1
    eval_config.num_steps = 5

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        if FLAGS.train:
            print('training')

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)   
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, inverseDictionary, train_data, m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, inverseDictionary, valid_data, tf.no_op())
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                saver.save(session, FLAGS.checkpoint_dir + '/model.ckpt', global_step=i + 1)
        else:
            print('testing')
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint file found")

        test_perplexity = run_epoch(session, mtest, inverseDictionary, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
    # print (FLAGS.checkpoint_dir)