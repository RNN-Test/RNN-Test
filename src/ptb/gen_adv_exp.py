from ptb_word_lm_wrapped import *
from coverage import *
import myreader as reader
import random
import matplotlib.pyplot as plt
from scipy.special import expit
import csv
import os
import collections

flags = tf.flags

flags.DEFINE_integer('topk', 3, 'topk')
flags.DEFINE_string('guided_coverage',
                    "DX", 'guided_coverage,could be "hidden_state","cell_state","DX",'
                                    '"input_gate","new_input_gate","forget_gate","output_gate"')
flags.DEFINE_integer('times_nn', 5, 'num_neurons = model.num_steps * times_of_ns')
flags.DEFINE_integer('objective', 1, '0 for objective1, 1 for objective2, 2 for objective1 + objective2')
flags.DEFINE_string('objective1', "diff", 'objective1 could be "diff", "cost", "state_diff"')
flags.DEFINE_integer('times_ns', 3, 'num_samples = model.batch_size * model.num_steps * times_ns')
flags.DEFINE_integer('lr_l', 0, "the lower bound of the learning rate")
flags.DEFINE_integer('lr_u', 70, 'the upper bound of the learning rate')
flags.DEFINE_string('exp', "adv", 'exp could be "baseline" and "adv"')

FLAGS = flags.FLAGS


def run_adv(session, model, inverseDictionary, data, eval_op, csv_filename, verbose=False):
    """Runs the model on the given experiment data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    costs = 0.0
    costs_adv = 0.0  # the cost for adv_sequences
    iters = 0
    start_time = time.time()
    state = session.run(model.initial_state)
    generate_num = 0
    csv_dict = []   # save the value that need to be saved to a csv file
    final_step = 0

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):
        print("------Step ", step, "-------")
        step_dict = collections.OrderedDict()
        step_dict['step'] = step

        guided_coverage = FLAGS.guided_coverage
        # for coverage
        states = model.states_all
        h_list = []
        c_list = []
        for s, state_tuple in enumerate(states):
            h_list.append([])
            c_list.append([])
            for l, (c, h) in enumerate(state_tuple):
                h_list[s].append(h)
                c_list[s].append(c)
        h_states_all = tf.stack(h_list)  # [step, layer, batch, embedding]
        c_states_all = tf.stack(c_list)

        ori_sequences = id_to_words(x, inverseDictionary)
        print("ori_se")
        print(ori_sequences)
        print("ori_sequences: ", ' '.join(ori_sequences[0]))
        next_batch_sequences = predict_next_batch(model, x, session, inverseDictionary)
        print('next_batch_sequences: ', ' '.join(next_batch_sequences[0]))
        num_samples = model.batch_size * model.num_steps * FLAGS.times_ns
        ori_samples = id_to_words(do_sample(session, model, x, num_samples), inverseDictionary)
        print("ori_samples: ", ' '.join(ori_samples[0]))

        # originally predict
        fetches = [model.cost, model.initial_state, model.inputs, model.embedding,
                   c_states_all, h_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, initial_state, inputs, embedding, c_states_all_value, h_states_all_value \
            = session.run(fetches, feed_dict)

        # update the coverage information after each prediction
        # update_coverage(h_states_all_value, c_states_all_value, inputs, initial_state, guided_coverage, step_dict, "ori")

        # which time step of the input to be modified
        step_index = random.randint(1, model.num_steps - 1)

        # generate adversarial examples
        adv_start = time.clock()
        if FLAGS.objective == 0:  # for diff
            objective1 = get_objective1(model, step_index, c_states_all, h_states_all)
            objective = objective1
        elif FLAGS.objective == 1:
            num_neurons = model.batch_size * model.num_steps * FLAGS.times_nn
            # print("num_neurons: ", num_neurons)
            objective2 = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all, h_states_all_value,
                                          c_states_all_value)
            # print("neurons_list: ", len(neurons_list))
            objective = objective2
        elif FLAGS.objective == 2:
            objective1 = get_objective1(model, step_index, c_states_all, h_states_all)
            num_neurons = model.batch_size * model.num_steps * FLAGS.times_nn
            objective2 = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all, h_states_all_value,
                                          c_states_all_value)
            objective = objective1 + objective2
        else:
            objective = 0
            print('Invalid input.  0 for objective1, 1 for objective2, 2 for objective1 + objective2')
        print("getting grads")
        grads = tf.gradients(objective, model.inputs)

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        grads_value = session.run(grads, feed_dict)

        x_prime = generate_adv_sequences(x, inputs, grads_value, embedding, step_index, step_dict)
        adv_sequences = id_to_words(x_prime, inverseDictionary)
        print('adv_sequences: ', ' '.join(adv_sequences[0]))

        adv_end = time.clock()
        duration = adv_end - adv_start
        step_dict['adv_time'] = duration

        # whether successfully generate adv
        is_generate = is_generate_adv(x, x_prime)
        print('is generate', is_generate)
        print("x: ", x, "x_prime: ", x_prime)
        step_dict['is_generate'] = is_generate
        if is_generate:
            generate_num += 1

        next_adv_batch_sequences = predict_next_batch(model, x_prime, session, inverseDictionary)
        print('next_adv_batch_sequences = ', ' '.join(next_adv_batch_sequences[0]))
        adv_samples = id_to_words(do_sample(session, model, x_prime, num_samples), inverseDictionary)
        print("samples after adversarial attacks: ", ' '.join(adv_samples[0]))

        # predict after modification
        fetches = [model.cost, model.initial_state, model.inputs, h_states_all, c_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x_prime
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost_adv, new_initial_state, new_inputs, h_states_all_adv, c_states_all_adv = \
            session.run(fetches, feed_dict)
        update_coverage(h_states_all_adv, c_states_all_adv, new_inputs, new_initial_state, guided_coverage, step_dict, "adv")

        # accumulate
        costs += cost
        costs_adv += cost_adv
        iters += model.num_steps

        # each step
        perplexity = np.exp(cost/model.num_steps)
        perplexity_adv = np.exp(cost_adv/model.num_steps)

        step_dict['perplexity'] = perplexity
        step_dict['perplexity_adv'] = perplexity_adv
        print("perplexity: ", perplexity)
        print("perplexity_adv:", perplexity_adv)
        print('current pp:', np.exp(costs_adv/iters))
        if verbose and step % (epoch_size // 10) == 10:
            print("Perplexity of the original input sequences:")
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
            print("Perplexity of the adversarial input sequences:")
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs_adv / iters),
                   iters * model.batch_size / (time.time() - start_time)))

        csv_dict.append(step_dict)
        final_step = step

    save_csv(csv_dict, csv_filename)
    print('save success')
    return np.exp(costs / iters), np.exp(costs_adv / iters), float(generate_num) / final_step


def run_ori(session, model, inverseDictionary, data, eval_op, csv_filename, verbose=False):
    """Runs the model on the given experiment data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    csv_dict = []  # save the value that need to be saved to a csv file

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):
        print("------Step ", step, "-------")
        step_dict = collections.OrderedDict()
        step_dict['step'] = step

        guided_coverage = FLAGS.guided_coverage
        # for coverage
        states = model.states_all
        h_list = []
        c_list = []
        for s, state_tuple in enumerate(states):
            h_list.append([])
            c_list.append([])
            for l, (c, h) in enumerate(state_tuple):
                h_list[s].append(h)
                c_list[s].append(c)
        h_states_all = tf.stack(h_list)  # [step, layer, batch, embedding]
        c_states_all = tf.stack(c_list)

        # ori_sequences = id_to_words(x, inverseDictionary)
        # print("ori_sequences: ", ori_sequences)
        # next_batch_sequences = predict_next_batch(model, x, session, inverseDictionary)
        # print('next_batch_sequences: ', next_batch_sequences)
        # num_samples = model.batch_size * model.num_steps * FLAGS.times_ns
        # ori_samples = id_to_words(do_sample(session, model, x, num_samples), inverseDictionary)
        # print("ori_samples: ", ori_samples)

        # originally predict
        fetches = [model.cost, model.initial_state, model.inputs, model.embedding,
                   c_states_all, h_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, initial_state, inputs, embedding, c_states_all_value, h_states_all_value \
            = session.run(fetches, feed_dict)
        # update the coverage information after each prediction
        update_coverage(h_states_all_value, c_states_all_value, inputs, initial_state, guided_coverage, step_dict,
                        "ori")
        costs += cost
        iters += model.num_steps

        # each step
        perplexity = np.exp(cost / model.num_steps)

        step_dict['perplexity'] = perplexity

        # if verbose and step % (epoch_size // 10) == 10:
        #     print("Perplexity of the original input sequences:")
        #     print("%.3f perplexity: %.3f speed: %.0f wps" %
        #           (step * 1.0 / epoch_size, np.exp(costs / iters),
        #            iters * model.batch_size / (time.time() - start_time)))
        #     print("Perplexity of the randomly replaced input sequences:")
        #     print("%.3f perplexity: %.3f speed: %.0f wps" %
        #           (step * 1.0 / epoch_size, np.exp(costs_replaced / iters),
        #            iters * model.batch_size / (time.time() - start_time)))

        csv_dict.append(step_dict)

    save_csv(csv_dict, csv_filename)
    print('save success')
    return np.exp(costs / iters)


def run_base(session, model, inverseDictionary, data, eval_op, csv_filename, verbose=False):
    """Runs the model on the given experiment data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    costs_replaced = 0.0  # the cost for replaced_sequences
    iters = 0
    state = session.run(model.initial_state)
    csv_dict = []   # save the value that need to be saved to a csv file

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):
        print("------Step ", step, "-------")
        step_dict = collections.OrderedDict()
        step_dict['step'] = step

        guided_coverage = FLAGS.guided_coverage
        # for coverage
        states = model.states_all
        h_list = []
        c_list = []
        for s, state_tuple in enumerate(states):
            h_list.append([])
            c_list.append([])
            for l, (c, h) in enumerate(state_tuple):
                h_list[s].append(h)
                c_list[s].append(c)
        h_states_all = tf.stack(h_list)  # [step, layer, batch, embedding]
        c_states_all = tf.stack(c_list)

        ori_sequences = id_to_words(x, inverseDictionary)
        print("ori_sequences: ", ori_sequences)
        next_batch_sequences = predict_next_batch(model, x, session, inverseDictionary)
        print('next_batch_sequences: ', next_batch_sequences)
        num_samples = model.batch_size * model.num_steps * FLAGS.times_ns
        ori_samples = id_to_words(do_sample(session, model, x, num_samples), inverseDictionary)
        print("ori_samples: ", ori_samples)

        # originally predict
        fetches = [model.cost, model.initial_state, model.inputs, model.embedding,
                   c_states_all, h_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, initial_state, inputs, embedding, c_states_all_value, h_states_all_value \
            = session.run(fetches, feed_dict)

        # update the coverage information after each prediction
        # update_coverage(h_states_all_value, c_states_all_value, inputs, initial_state, guided_coverage, step_dict, "ori")

        # which time step of the input to be modified
        step_index = random.randint(1, model.num_steps - 1)

        # baseline
        x_random = generate_replaced_sequence(x, step_index)  # 1 is the amount of words wants to be replaced
        replaced_sequence = id_to_words(x_random, inverseDictionary)
        print("randomly replaced_sequence: ", ' '.join(replaced_sequence[0]))
        next_replaced_batch_sequences = predict_next_batch(model, x_random, session, inverseDictionary)
        print('next_replaced_batch_sequences', ' '.join(next_replaced_batch_sequences[0]))

        fetches = [model.cost, model.initial_state, model.inputs,
                   c_states_all, h_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x_random
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost_replaced, initial_state, inputs_random, c_states_all_random, h_states_all_random \
            = session.run(fetches, feed_dict)
        # update the coverage information after each prediction ....
        update_coverage(h_states_all_random, c_states_all_random, inputs_random, initial_state, guided_coverage, step_dict, "base")

        # accumulate
        costs += cost
        costs_replaced += cost_replaced
        iters += model.num_steps

        # each step
        perplexity = np.exp(cost/model.num_steps)
        perplexity_replaced = np.exp(cost_replaced/model.num_steps)

        step_dict['perplexity'] = perplexity
        step_dict['perplexity_adv'] = perplexity_replaced
        print("perplexity: ", perplexity)
        print("perplexity_replaced: ", perplexity_replaced)

        if verbose and step % (epoch_size // 10) == 10:
            print("Perplexity of the original input sequences:")
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
            print("Perplexity of the randomly replaced input sequences:")
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs_replaced / iters),
                   iters * model.batch_size / (time.time() - start_time)))

        csv_dict.append(step_dict)

    save_csv(csv_dict, csv_filename)
    print('save success')
    return np.exp(costs / iters), np.exp(costs_replaced / iters)


def update_coverage(h_states_all_value, c_states_all_value, inputs, initial_state, guided_coverage, step_dict, label):
    # compute the state coverage
    if guided_coverage == "hidden_state":
        hidden_state_c = hidden_state_coverage(h_states_all_value)
        if label == "ori":
            step_dict['hidden_state_c_ori'] = str(hidden_state_c)
            print("hidden_state_c_0:", hidden_state_c)
        elif label == "base":
            step_dict['hidden_state_c_base'] = str(hidden_state_c)
        elif label == "adv":
            step_dict['hidden_state_c_adv'] = str(hidden_state_c)
            print("hidden_state_c_1:", hidden_state_c)
    elif guided_coverage == "cell_state":
        cell_state_c = cell_state_coverage(c_states_all_value)
        if label == "ori":
            print("cell_state_c_ori:", cell_state_c)
            save_coverage('cell_state_c_ori',cell_state_c,step_dict)
        elif label == "base":
            print("cell_state_c_base:", cell_state_c)
            save_coverage('cell_state_c_base', cell_state_c,step_dict)
        elif label == "adv":
            save_coverage('cell_state_c_adv', cell_state_c, step_dict)
            print("cell_state_c_adv:", cell_state_c)
    elif "gate" in guided_coverage:
        # compute the gate coverage
        step, layer, batch, embedding = np.array(h_states_all_value).shape
        gates_all = np.array(h_states_all_value).swapaxes(0, 2)
        gates_all = gates_all.swapaxes(0, 1)
        gates_all = np.append(gates_all, [inputs], axis=0)  # add the input layer

        initial_array = []
        for layer_inital in initial_state:
            initial_array.append([layer_inital.h])  # add the initial states
        initial_array = np.append(np.array(initial_array), np.zeros((1, 1, batch, embedding)), axis=0)
        initial_array = initial_array.swapaxes(1, 2)

        gates_all = np.append(initial_array, gates_all, axis=2)  # shape: (layer+1, batch, step+1, embedding)

        input_gate_c, new_input_c, forget_gate_c, output_gate_c = gate_coverage(gates_all)

        if label == "ori":
            print("input_gate_c_ori:", input_gate_c)
            print("new_input_c_ori:", new_input_c)
            print("forget_gate_c_ori:", forget_gate_c)
            print("output_gate_c_ori:", output_gate_c)
            save_coverage('input_gate_c_ori',input_gate_c,step_dict)
            save_coverage('new_input_c_ori', new_input_c, step_dict)
            save_coverage('forget_gate_c_ori', forget_gate_c, step_dict)
            save_coverage('output_gate_c_ori', output_gate_c, step_dict)
        elif label == "base":
            print("input_gate_c_base:", input_gate_c)
            print("new_input_c_base:", new_input_c)
            print("forget_gate_c_base:", forget_gate_c)
            print("output_gate_c_base:", output_gate_c)
            save_coverage('input_gate_c_base', input_gate_c, step_dict)
            save_coverage('new_input_c_base', new_input_c, step_dict)
            save_coverage('forget_gate_c_base', forget_gate_c, step_dict)
            save_coverage('output_gate_c_base', output_gate_c, step_dict)
        elif label == "adv":
            print("input_gate_c_adv:", input_gate_c)
            print("new_input_c_adv:", new_input_c)
            print("forget_gate_c_adv:", forget_gate_c)
            print("output_gate_c_adv:", output_gate_c)
            save_coverage('input_gate_c_adv', input_gate_c, step_dict)
            save_coverage('new_input_c_adv', new_input_c, step_dict)
            save_coverage('forget_gate_c_adv', forget_gate_c, step_dict)
            save_coverage('output_gate_c_adv', output_gate_c, step_dict)
    elif guided_coverage == "DX":
        # compute the coverage like DeepXplore
        h_array = h_states_all_value.swapaxes(0, 2)  # batch, layer, step, embedding
        unit_coverage = unit_neuron_coverage(h_array)
        # embedding_coverage = embedding_neuron_coverage(h_array)
        print("unit_coverage:", unit_coverage)
        if label == "ori":
            print("embedding_coverage_ori:", unit_coverage)
            step_dict['embedding_coverage_ori'] = unit_coverage
        elif label == "base":
            print("embedding_coverage_base:", unit_coverage)
            step_dict['embedding_coverage_base'] = unit_coverage
        elif label == "adv":
            print("embedding_coverage_adv:", unit_coverage)
            step_dict['embedding_coverage_adv'] = unit_coverage
    else:
        print("None coverage criteria named: ", guided_coverage)


def get_objective1(model, step, h_states_all, c_states_all):
    objective1 = 0
    if FLAGS.objective1 == "diff":
        topk = FLAGS.topk
        temp_index = step
        while temp_index < model.batch_size * model.num_steps:
            topk_labels = tf.nn.top_k(model.logits[temp_index, ...], k=topk, sorted=True).indices
            # increase the probability of other labels
            for i in range(1, topk):
                objective1 += model.logits[temp_index, topk_labels[i]]
            # decrease the probability of the original label
            objective1 -= model.logits[temp_index, topk_labels[0]]
            temp_index += model.num_steps

    elif FLAGS.objective1 == "cost":
        objective1 = model.cost
    elif FLAGS.objective1 == "state_diff":
        objective1 = h_states_all[step - 1] + c_states_all[step] - h_states_all[step]  # state diff
    else:
        print("Objective1 not configured.")
    return objective1


def save_coverage(name, coverages, step_dict):
    i = 0
    for item in coverages:
        step_dict[name+'['+str(i)+']'] = item
        i += 1


def predict_next_batch(model, sequence, session, inverseDictionary):
    batch_size, num_steps = np.array(sequence).shape
    next_batch_sequences = []
    state = session.run(model.initial_state)

    steps = num_steps
    while steps > 0:
        fetches = [model.initial_state, model.logits]
        feed_dict = {}
        feed_dict[model.input_data] = sequence

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        initial_state, logits_value = session.run(fetches, feed_dict)
        decoded_y = predict_raw(logits_value, sequence)

        last_ids = np.array(decoded_y)[:, -1].reshape(batch_size, 1)
        last_sequences = id_to_words(last_ids, inverseDictionary)

        next_sequence = np.append(sequence[:, 1:], last_ids, axis=1)
        sequence = next_sequence

        if len(next_batch_sequences) <= 0:
            next_batch_sequences = last_sequences
        else:
            next_batch_sequences = np.append(next_batch_sequences, last_sequences, axis=1)

        steps -= 1
    return next_batch_sequences


def do_sample(session, model, seeds, num_samples):
    """Sampled from the model"""
    batch_size, num_steps = np.array(seeds).shape
    samples = []
    state = session.run(model.initial_state)
    batch_num_samples = num_samples / model.batch_size

    cur_seeds = seeds
    while batch_num_samples > 0:

        fetches = [model.initial_state, model.sample]
        feed_dict = {}
        feed_dict[model.input_data] = cur_seeds

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        initial_state, sample = session.run(fetches, feed_dict)

        sample = sample.reshape(batch_size, num_steps)
        last_sample = sample[:, -1].reshape(batch_size, 1)

        next_seeds = np.append(cur_seeds[:, 1:], last_sample, axis=1)
        cur_seeds = next_seeds

        if len(samples) <= 0:
            samples = last_sample
        else:
            samples = np.append(samples, last_sample, axis=1)

        batch_num_samples -= 1
    return samples


# decode logits to words
def predict_raw(logits_value, x):
    batch_size, num_steps = np.array(x).shape

    logits = np.exp(logits_value) / sum(np.exp(logits_value))
    y_hat_sequences = []
    y_hat_sequence = []

    for step_index, each_logits in enumerate(logits):
        decoded_word_id = int(np.argmax(each_logits))
        y_hat_sequence.append(decoded_word_id)

        if (step_index + 1) % num_steps == 0:
            y_hat_sequences.append(y_hat_sequence)
            y_hat_sequence = []

    return y_hat_sequences


def predict(logits_value, x, step_to_modify, inverseDictionary):
    batch_size, num_steps = np.array(x).shape
    logits = np.exp(logits_value) / sum(np.exp(logits_value))  # [batch_size * num_steps, vocab_size]

    y_hat_sequences = []
    y_hat_sequence = []
    for step_index, each_logits in enumerate(logits):
        # show the actual prediction
        decodedWordId = int(np.argmax(each_logits))
        y_hat_sequence.append(decodedWordId)

        row_index = step_index / num_steps
        column_index = step_index % num_steps
        # plot the probabilities
        if column_index == step_to_modify:
            idx_sort = np.argsort(each_logits)[::-1]
            word_sort = [inverseDictionary[int(x1)] for x1 in idx_sort]
            prob_sort = [each_logits[x1] for x1 in idx_sort]
            #plot_bar(word_sort[:10], prob_sort[:10], inverseDictionary[x[row_index, column_index]])
        # start another batch
        if (step_index + 1) % num_steps == 0:
            y_hat_sequences.append(y_hat_sequence)
            y_hat_sequence = []

    return y_hat_sequences


def id_to_words(id_sequence, inverseDictionary):
    words_sequences = []
    batch_size, num_steps = np.array(id_sequence).shape
    for batch_id in range(batch_size):
        each_sequence = []
        for step_id in range(num_steps):
            each_sequence.append(inverseDictionary[id_sequence[batch_id][step_id]])
        words_sequences.append(each_sequence)
    return words_sequences


def plot_bar(word_list, prob_list, title):
    rects = plt.bar(range(len(prob_list)), prob_list, color='b')
    # x legend
    index = range(len(word_list))
    index = [float(c) + 0.4 for c in index]
    plt.ylim(ymax=max(prob_list)+0.01, ymin=0)
    plt.xticks(index, word_list)
    plt.ylabel("probability")  # y legend
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height,3)), ha='center', va='bottom')
    # plt.show()
    if '\\' in title:
        return
    plt.title("Top 10 prediction probability for ["+title+']')
    plt.savefig('./figures/' + title+'.png')
    plt.close()


def generate_adv_sequences(x, inputs, grads_value, embedding, step_index,step_dict):
    batch_size, num_steps, _ = np.array(inputs).shape   #1,10
    adv_sequences = np.copy(x)
    pert = grads_value[0]

    step_dict['perturbation'] = 0

    for input_index in range(batch_size):
        # just apply the perturb to the word of the step_index
        step_pert = np.array(pert[input_index][step_index]).flatten()
        # step_pert = step_pert / np.linalg.norm(step_pert)
        step_input = np.array(inputs[input_index][step_index]).flatten()

        last_id = x[input_index][step_index]
        lr_value = 0

        for learning_rate in range(FLAGS.lr_l, FLAGS.lr_u, 5):
            # print("learning_rate:", learning_rate)
            pert_val = step_pert * learning_rate
            new_input_val = step_input + pert_val
            embedding_val = np.array(embedding)

            lr_value = learning_rate
            dist = []
            for emb in embedding_val:
                dist.append(np.linalg.norm(emb - new_input_val))
            min_id = int(np.argmin(dist))  # the word that is nearest to the new input embedding

            if last_id != min_id:
                step_dict['perturbation'] = np.linalg.norm(pert_val)
                adv_sequences[input_index][step_index] = min_id
                break
        step_dict['learning_rate'] = lr_value
    return adv_sequences


def generate_replaced_sequence(x, step_index):
    batch, num_steps = np.array(x).shape
    if step_index > num_steps:
        print('The step to be replaced exceeds the total number, so the function will return the original sequence.')
        return x
    replaced_sequence = np.copy(x)
    for b in range(batch):
        wordId = random.randint(0, 10000)  # vocab_size = 10000, could be passed by parameter
        replaced_sequence[b][step_index] = wordId
    return replaced_sequence


# to judge if adv has been generated
def is_generate_adv(x, y):
    result = (x == y)
    if False in result:
        return True
    else:
        return False


def save_csv(dict, filename):
    if os.path.exists('./experiment_result'):
        pass
    else:
        os.makedirs('./experiment_result')
    filename = './experiment_result/'+filename
    with open(filename, 'wb') as f:
        w = csv.writer(f)
        fields_names = dict[0].keys()
        w.writerow(fields_names)
        for row in dict:
            w.writerow(row.values())


def test():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB experimentdata directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _, word_to_id = raw_data
    # Rani: added inverseDictionary
    inverseDictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    config = get_config()
    eval_config = get_config()
    # print(eval_config.batch_size, eval_config.num_steps)
    eval_config.batch_size = 1
    eval_config.num_steps = 10

    # csv_filename = 'data'+str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))+'.csv'
    if FLAGS.exp == "baseline":
        csv_filename = str(FLAGS.guided_coverage) + "_baseline_" + str(int(time.time())) + ".csv"
    elif FLAGS.objective == 1:
        csv_filename = str(FLAGS.objective) + "_" + str(FLAGS.guided_coverage) + "_adv_" + str(
        int(time.time())) + ".csv"
    else:
        csv_filename = str(FLAGS.objective) + "_" + str(FLAGS.objective1) + "_" + str(
            FLAGS.guided_coverage) + "_adv_" + str(int(time.time())) + ".csv"
    print("csv_filename=", csv_filename)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # necessary to run
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()
        tf.initialize_all_variables().run()

        print('testing')
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")

        # csv_filename_ori = str(FLAGS.guided_coverage) + "_ori_" + ".csv"
        # if csv_filename_ori not in os.listdir("./experiment_result/"):
        #     test_perplexity_ori = run_ori(session, mtest, inverseDictionary, test_data, tf.no_op(),
        #                               csv_filename=csv_filename_ori)
        #     with open('./experiment_result/'+csv_filename_ori, 'a+') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(['Original Test Perplexity', test_perplexity_ori])
        #     exit(0)
            
        start = time.clock()
        if FLAGS.exp == "baseline":
            test_perplexity, test_perplexity_random = run_base(session, mtest, inverseDictionary,
                                                                                   test_data, tf.no_op(),
                                                                                   csv_filename=csv_filename)
        else:
            test_perplexity, test_perplexity_adv, generate_rate = run_adv(session, mtest, inverseDictionary,
                                                                                    test_data, tf.no_op(),
                                                                                    csv_filename=csv_filename)

        end = time.clock()
        duration = end-start
        print("Run time: ", duration)
        # print("Original Test Perplexity: %.3f" % test_perplexity)
        # print("Test Perplexity after adversarial attacks: %.3f" % test_perplexity_adv)

        with open('./experiment_result/'+csv_filename, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(['Total Run Time', duration])
            writer.writerow(['Original Test Perplexity', test_perplexity])
            if FLAGS.exp == "baseline":
                writer.writerow(['Test Perplexity of random replacement', test_perplexity_random])
            else:
                writer.writerow(['Test Perplexity after adversarial attacks', test_perplexity_adv])
                writer.writerow(['Generation Rate', generate_rate])

        print('save '+csv_filename + 'successfully')


if __name__ == "__main__":
    test()
