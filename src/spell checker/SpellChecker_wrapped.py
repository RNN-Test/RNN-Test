# # Creating a Spell Checker

# The objective of this project is to build a model that can take a sentence with spelling mistakes as input,
#  and output the same sentence, but with the mistakes corrected. The data that we will use for this project
# will be twenty popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads).
# Our model is designed using grid search to find the optimal architecture, and hyperparameter values. The best results,
#  as measured by sequence loss with 15% of our data, were created using a two-layered network with a bi-direction RNN
# in the encoding layer and Bahdanau Attention in the decoding layer. [FloydHub's](https://www.floydhub.com/) GPU
# service was used to train the model.
# 
# The sections of the project are:
# - Loading the Data
# - Preparing the Data
# - Building the Model
# - Training the Model
# - Fixing Custom Sentences
# - Summary

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from statefulRNNCell import StatefulRNNCell


# The default parameters
epochs = 100
batch_size = 2  # set as 128 before
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75

# Parameters used in gen_adv.py
max_length = 92
min_length = 10

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]


# ## Loading the Data
def load_book(path):
    """Load a book from its file"""
    input_file = os.path.join(path)
    with open(input_file) as f:
        book = f.read()
    return book


def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


def model_inputs():
    '''Create palceholders for inputs to the model'''
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    '''Create the encoding layer'''
    # records to compute the coverage
    layers_enc_outputs = []
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, 
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state

    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(num_units=rnn_size, state_is_tuple=False)
                    cell_fw = StatefulRNNCell(cell_fw)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                            input_keep_prob=keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(num_units=rnn_size, state_is_tuple=False)
                    cell_bw = StatefulRNNCell(cell_bw)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                            input_keep_prob=keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
                    layers_enc_outputs.append(enc_output)  # record the data for computing the coverage

                    enc_output_h = []
                    for output in enc_output:
                        output_c, output_h = tf.split(output, num_or_size_splits=2, axis=2)
                        enc_output_h.append(output_h)
                    enc_output_h = tuple(enc_output_h)
                    enc_output = enc_output_h
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output, 2)  # construct the same output as

            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0], layers_enc_outputs


def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_target_length):
    '''Create the training logits'''
    
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_target_length)

        # training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
        #                                                           output_time_major=False,
        #                                                           impute_finished=True,
        #                                                           maximum_iterations=max_target_length)

        # print('training_helper = ', training_helper)
        # print('training_decoder = ', training_decoder)
        # print('training_logits = ', training_logits)
        # print('final_state = ', final_state)
        # print('final_sequence_length = ', final_sequence_length)

        return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):
    '''Create the inference logits'''
    
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)

        # print('start_tokens = ', start_tokens)
        # print('inference_helper = ', inference_helper)
        # print('inference_decoder = ', inference_decoder)
        # print('inference_logits = ', inference_logits)
        # print('final_state = ', final_state)
        # print('final_sequence_length = ', final_sequence_length)

        return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, 
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                # cell = tf.contrib.rnn.LSTMCell(num_units=rnn_size, state_is_tuple=False)
                # cell = StatefulRNNCell(cell)

                lstm = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=False)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                              attn_mech,
                                                              rnn_size)

    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state)

    # with tf.name_scope("Attention_Wrapper"):
    #     dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
    #                                                           attn_mech,
    #                                                           rnn_size)
    #
    # initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state,
    #                                                                 _zero_state_tensors(rnn_size,
    #                                                                                     batch_size,
    #                                                                                     tf.float32))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  targets_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):
    '''Use the previous functions to create the training and inference logits'''

    # return vocab_size * embedding_size martrix, value betwenn -1 and 1
    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    # select elements with index(inputs(int32)) in enc_embeddings
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state, layers_enc_outputs = encoding_layer(rnn_size, inputs_length, num_layers,
                                           enc_embed_input, keep_prob, direction)
    
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # print('enc_embeddings',enc_embeddings)
    # print('enc_embed_input',enc_embed_input)
    # print('enc_output',enc_output)
    # print('enc_state',enc_state)
    # print('dec_embeddings',dec_embeddings)
    # print('dec_input',dec_input)
    # print('dec_embed_input',dec_embed_input)

    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state,
                                                        vocab_size, 
                                                        inputs_length, 
                                                        targets_length, 
                                                        max_target_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers,
                                                        direction)
    
    return training_logits, inference_logits, enc_embed_input, enc_embeddings, layers_enc_outputs


def noise_maker(sentence, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''

    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0, 1, 1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0, 1, 1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i + 1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass
        i += 1
    return noisy_sentence


def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):
    tf.reset_default_graph()

    # Load the model inputs
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits, input_embedding, inputs_all_embeddings, layers_enc_outputs = \
        seq2seq_model(tf.reverse(inputs, [-1]),targets,
                                               keep_prob,
                                               inputs_length,
                                               targets_length,
                                               max_target_length,
                                               len(vocab_to_int) + 1,
                                               rnn_size,
                                               num_layers,
                                               vocab_to_int,
                                               batch_size,
                                               embedding_size,
                                               direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)
        rnn_outputs = inference_logits.rnn_output

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')

    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                targets,
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # _reverse function in bidirectional_dynamic_rnn implementation
    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        return array_ops.reverse(input_, axis=[seq_dim])
    with vs.variable_scope("bw") as bw_scope:
        time_major = False
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1
        input_reverse = _reverse(input_embedding, seq_lengths=inputs_length, seq_dim=time_dim, batch_dim=batch_dim)

    # Merge all of the summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length', 'predictions',
                    'merged', 'train_op', 'optimizer',
                    'input_embedding', 'input_reverse', 'inputs_all_embeddings', 'rnn_outputs', 'layers_enc_outputs']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def text_to_ints(text):
    '''Prepare the text for the model'''
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]


# Process the data
path = './books/'
book_files = [f for f in listdir(path) if isfile(join(path, f))]
book_files = book_files[1:]

books = []
for book in book_files:
    books.append(load_book(path + book))

clean_books = []
for book in books:
    clean_books.append(clean_text(book))

vocab_to_int = {}
count = 0
for book in clean_books:
    for character in book:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

# Add special tokens to vocab_to_int
codes = ['<PAD>', '<EOS>', '<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1

vocab_size = len(vocab_to_int)
print("The vocabulary contains {} characters.".format(vocab_size))
print(sorted(vocab_to_int))

int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character

# I hope that you have found this project to be rather interesting and useful. The example sentences that I
# have presented above were specifically chosen, and the model will not always be able to make corrections
# of this quality. Given the amount of data that we are working with, this model still struggles. For it to
#  be more useful, it would require far more training data, and additional parameter tuning. This parameter
#  values that I have above worked best for me, but I expect there are even better values that I was not able to find.
#
# Thanks for reading!
