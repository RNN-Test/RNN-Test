"""
Coverage Criteria for the Testing of SpellChecker
include both ours and those of DeepXplore
"""

import numpy as np
from scipy.special import expit
import tensorflow as tf
from SpellChecker_wrapped import embedding_size, rnn_size
from nltk.translate.bleu_score import sentence_bleu

hidden_state_records = []
cell_state_records = []

input_gate_records = []
forget_gate_records = []
output_gate_records = []
new_input_gate_records = []

sigmoid_sections_rec = [0.3, 0.4, 0.6, 0.7]
tanh_sections_rec = [-0.7, -0.2, 0.2, 0.7]

# < the lower bound & > the upper bound are the two boundary sections
sigmoid_sections = [0.1, 0.4, 0.6, 0.9]  # according to the distribution of sigmoid function
tanh_sections = [-0.8, -0.2, 0.2, 0.8]  # according to the distribution of tanh function

# threshold_hidden = 0.01
threshold_deepXplore = 0.5
embedding_records = []
unit_records = []


# ---------------------------------#
# Coverage Criteria of ours
def hidden_state_coverage(h_states_all_value, max_sen_length=None):
    layers, fw_bw, batch, steps, r_size = np.array(h_states_all_value).shape
    h_all = h_states_all_value.flatten()
    h_all = np.reshape(h_all, [-1, steps, r_size])

    global hidden_state_records
    if hidden_state_records == []:
        hidden_state_records = np.zeros([layers * fw_bw * batch, max_sen_length, r_size])

    if steps > max_sen_length:
        # h_all = h_all[:, 0:steps, :]
        h_all = h_all[:, 0:max_sen_length, :] 

    top_neurons = 0
    for s, s_array in enumerate(h_all):
        for r, r_array in enumerate(s_array):
            neuron_id = np.argmax(r_array)
            hidden_state_records[s][r][neuron_id] = 1  # update

    h_flatten = hidden_state_records.flatten()
    for neuron in h_flatten:
        if neuron == 1:  # record
            top_neurons += 1

    hidden_state_c = float(top_neurons) / (layers * fw_bw * batch * max_sen_length * r_size)
    # hidden_state_c *= max_sen_length / steps  # maybe wrong
    return hidden_state_c


# slow
def cell_state_coverage(c_states_all_value, max_sen_length=None):  # use the tanh sections
    layers, fw_bw, batch, steps, r_size = np.array(c_states_all_value).shape
    sections = len(tanh_sections) + 1

    c_all = c_states_all_value.flatten()
    c_all = np.reshape(c_all, [-1, steps, r_size])

    global cell_state_records
    if cell_state_records == []:
        cell_state_records = np.zeros([layers * fw_bw * batch, max_sen_length, r_size, sections])

    if steps > max_sen_length:
        c_all = c_all[:, 0:steps, :]
    cell_state_c = np.zeros(sections, dtype=float)

    for s, s_array in enumerate(c_all):
        for r, r_array in enumerate(s_array):
            r_array = np.tanh(r_array)
            for n, neuron in enumerate(r_array):
                # update
                isset = False
                for sec, sec_value in enumerate(tanh_sections):
                    if neuron < sec_value:
                        cell_state_records[s][r][n][sec] = 1
                        isset = True
                        break
                if not isset:
                    cell_state_records[s][r][n][sections - 1] = 1

    c_flatten = cell_state_records.flatten()
    c_flatten = c_flatten.reshape([-1, sections])
    for r, r_array in enumerate(c_flatten):
        for sec in range(len(r_array)):
            if c_flatten[r][sec] == 1:
                cell_state_c[sec] += 1

    cell_state_c /= layers * fw_bw * batch * max_sen_length * r_size
    # cell_state_c *= max_sen_length / steps
    return cell_state_c


# gate coverage not presented in RNN-Test paper
def gate_coverage(gates_all, inputs, max_sen_length=None):
    layers, fw_bw, batch, steps, r_size = np.array(gates_all).shape  # layers, fw_bw, batch, steps+1, rnn_size
    _, _, _, embedding = np.array(inputs).shape

    gates_all = gates_all.swapaxes(1, 3)  # layers, steps+1, batch, fw_bw, rnn_size
    inputs = inputs.swapaxes(0, 2)  # steps, batch, fw_bw, embedding

    units = (embedding + r_size) // 4
    # store the values of each time
    # input_gate_all = np.zeros([layers, steps - 1, batch, fw_bw, units])
    input_gate_all = np.zeros([steps - 1, layers, batch, fw_bw, units])
    forget_gate_all = np.zeros_like(input_gate_all)
    output_gate_all = np.zeros_like(input_gate_all)
    new_input_gate_all = np.zeros_like(input_gate_all)

    # construct the data for each gate
    for s in range(1, steps):  # the loop starts from layer 0, without the inputs as the layer 0
        for l in range(0, layers):
            # according to the computation of each lstm cell, concatenate inputs and h and split to four parts
            concat = np.concatenate((inputs[s-1], gates_all[l][s-1]), axis=2)
            input_gates, new_input_gates, forget_gates, output_gates = np.split(concat, 4, axis=2)

            l_index = l
            s_index = s-1  # the record index starts from 0
            # Also corresponds to the activation functions each gate uses
            input_gate_all[s_index][l_index] = expit(input_gates)  # expit: sigmoid
            new_input_gate_all[s_index][l_index] = np.tanh(new_input_gates)
            forget_gate_all[s_index][l_index] = expit(forget_gates)
            output_gate_all[s_index][l_index] = expit(output_gates)

    # update the coverage
    global input_gate_records, forget_gate_records, new_input_gate_records, output_gate_records
    activation_f = ["sigmoid", "tanh", "sigmoid", "sigmoid"]  # could be configured
    sections_num = [0, 0, 0, 0]
    for index, ac_f in enumerate(activation_f):
        if ac_f == "sigmoid":
            sections_num[index] = len(sigmoid_sections) + 1
        elif ac_f == "tanh":
            sections_num[index] = len(tanh_sections) + 1

    # initialization
    if len(input_gate_records) <= 0:
        e_num = layers * batch * fw_bw * units
        input_gate_records = np.zeros([max_sen_length, e_num, sections_num[0]])
        forget_gate_records = np.zeros([max_sen_length, e_num, sections_num[1]])
        new_input_gate_records = np.zeros([max_sen_length, e_num, sections_num[2]])
        output_gate_records = np.zeros([max_sen_length, e_num, sections_num[3]])

    input_gate_c = each_gate_coverage(input_gate_all, activation_f[0], input_gate_records, max_sen_length)
    new_input_c = each_gate_coverage(new_input_gate_all, activation_f[1], new_input_gate_records, max_sen_length)
    forget_gate_c = each_gate_coverage(forget_gate_all, activation_f[2], forget_gate_records, max_sen_length)
    output_gate_c = each_gate_coverage(output_gate_all, activation_f[3], output_gate_records, max_sen_length)

    return input_gate_c, new_input_c, forget_gate_c, output_gate_c


def each_gate_coverage(gate_array, activation_f, records, max_sen_length=None):
    steps, layers, batch, fw_bw, units = gate_array.shape
    sections_num = records.shape[-1]
    if activation_f == "sigmoid":
        sections = sigmoid_sections_rec
    else:
        sections = tanh_sections_rec

    gate_c = np.zeros(sections_num, dtype=float)

    gate_all = gate_array.flatten()
    gate_all = np.reshape(gate_all, [steps, -1])
    for s, s_array in enumerate(gate_all):
        for r, r_value in enumerate(s_array):
            # update
            isset = False
            for sec, sec_value in enumerate(sections):
                if r_value < sec_value:
                    records[s][r][sec] = 1
                    isset = True
                    break
            if not isset:
                records[s][r][sections_num - 1] = 1

    # record
    g_flatten = records.flatten()
    g_flatten = g_flatten.reshape([-1, sections_num])
    for r, r_array in enumerate(g_flatten):
        for sec in range(len(r_array)):
            if g_flatten[r][sec] == 1:
                gate_c[sec] += 1

    gate_c /= layers * max_sen_length * batch * fw_bw * units
    return gate_c


# ---------------------------------#
# coverage criteria like DeepXplore
def unit_neuron_coverage(h_states_all_value, max_sen_length=None):
    batch, layers, steps, fw_bw, embedding = np.array(h_states_all_value).shape
    if steps > max_sen_length:
        h_states_all_value = h_states_all_value[:, :, 0:steps, :, :]

    h_reshape = np.reshape(h_states_all_value, [batch, layers, steps, -1])
    # print("h_reshape.shape")
    # print(h_reshape.shape)
    h_mean = np.mean(h_reshape, axis=3)
    # print("h_mean shape")
    # print(h_mean.shape)
    # h_all = h_mean.flatten()
    # h_all = np.reshape(h_all, [batch, layers, steps, fw_bw * embedding])

    global unit_records
    if len(unit_records) <= 0:
        unit_records = np.zeros([batch, layers, max_sen_length])

    print("unit_records shape")
    print(unit_records.shape)

    neurons_covered = 0
    for b, batch_array in enumerate(h_mean):
        for l, layer_array in enumerate(batch_array):
            scaled_layer = scale(np.array(layer_array))
            for u, unit in enumerate(scaled_layer):
                if unit >= threshold_deepXplore:
                    unit_records[b][l][u] = 1  # covered

    e_flatten = unit_records.flatten()
    for neuron in e_flatten:
        if neuron > 0:
            neurons_covered += 1

    all_neurons = batch * layers * max_sen_length
    coverage = float(neurons_covered) / all_neurons
    # coverage *= max_sen_length / steps
    return coverage


# necessary functions
def scale(output, max=1, min=0):
    std = (output - output.min()) / (output.max() - output.min())
    scaled = std * (max - min) + min
    return scaled


# select states to boost, default way
def neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all, h_states_all_value, c_states_all_value,
                     seq_length):
    neurons_tensor = 0
    neurons_list = []
    print("neuron selection")
    if guided_coverage == "hidden_state":
        layers, fw_bw, batch, steps, r_size = np.array(h_states_all_value).shape
        top_neurons = steps * layers * batch * fw_bw
        # top_neurons = int(steps * layers * batch * embedding * threshold_hidden)
        h_all = h_states_all_value.flatten()

        start = 0
        h_array_sorted = sorted(h_all)
        for i, e_value in enumerate(h_array_sorted):
            if e_value > 0.3:
                start = i
                break
                # neuron_ids = np.array(h_all).argsort()[-(top_neurons + num_neurons): -top_neurons]  # larger values
        neuron_ids = np.array(h_all).argsort()[start: start+num_neurons]  # smaller values

        h_states_tensors = tf.reshape(h_states_all, [-1])  # flatten the tensor
        for nid in neuron_ids:
            neurons_tensor += h_states_tensors[nid]
            neurons_list.append(h_states_tensors[nid])

    elif guided_coverage == "cell_state":
        lower_neurons = num_neurons // 2
        higher_neurons = num_neurons - lower_neurons
        c_all = c_states_all_value.flatten()

        c_tanh = np.tanh(c_all)
        e_array_sorted = sorted(c_tanh)
        e_array_sorted_reverse = sorted(c_tanh, reverse=True)

        low_start, high_start = 0, 0
        for i, e_value in enumerate(e_array_sorted):
            if e_value > tanh_sections[0]:
                low_start = i
                break
        for i, e_value in enumerate(e_array_sorted_reverse):
            if e_value < tanh_sections[len(tanh_sections) - 1]:
                high_start = i
                break

        if high_start > 0:
            higher_neuron_ids = np.array(c_tanh).argsort()[-(high_start + higher_neurons):-high_start]
        else:
            higher_neuron_ids = np.array(c_tanh).argsort()[-higher_neurons:]

        lower_neuron_ids = np.array(c_tanh).argsort()[low_start:(low_start + lower_neurons)]

        c_states_tensors = tf.reshape(c_states_all, [-1])
        for ln_id in lower_neuron_ids:  # decrease the value of lower_neurons
            neurons_tensor -= c_states_tensors[ln_id]
            neurons_list.append(c_states_tensors[ln_id])
        for hn_id in higher_neuron_ids:  # increase the value of higher_neurons
            neurons_tensor += c_states_tensors[hn_id]
            neurons_list.append(c_states_tensors[hn_id])

    elif guided_coverage in ["forget_gate", "output_gate"]:
        lower_neurons = num_neurons // 2
        higher_neurons = num_neurons - lower_neurons

        h_all = h_states_all_value.flatten()

        e_array = expit(h_all)
        e_array_sorted = sorted(e_array)
        e_array_sorted_reverse = sorted(e_array, reverse=True)

        low_start, high_start = 0, 0
        for i, e_value in enumerate(e_array_sorted):
            if e_value > sigmoid_sections[0]:
                low_start = i
                break
        for i, e_value in enumerate(e_array_sorted_reverse):
            if e_value < sigmoid_sections[len(sigmoid_sections) - 1]:
                high_start = i
                break
        if high_start > 0:
            higher_neuron_ids = np.array(e_array).argsort()[-(high_start + higher_neurons):-high_start]
        else:
            higher_neuron_ids = np.array(e_array).argsort()[-higher_neurons:]
        lower_neuron_ids = np.array(e_array).argsort()[low_start:(low_start + lower_neurons)]

        h_states_tensors = tf.reshape(h_states_all, [-1])
        for index, ln_id in enumerate(lower_neuron_ids):
            tensor_id_value = ln_id
            neurons_tensor -= h_states_tensors[tensor_id_value]
            neurons_list.append(h_states_tensors[tensor_id_value])
        for index, hn_id in enumerate(higher_neuron_ids):
            tensor_id_value = hn_id
            neurons_tensor += h_states_tensors[tensor_id_value]
            neurons_list.append(h_states_tensors[tensor_id_value])

    elif guided_coverage == "DX":
        if len(unit_records) <= 0:
            return h_states_all[0][0][0][0][0], []

        layers, fw_bw, batch, steps, rnn_size = np.array(h_states_all_value).shape

        unit_records_temp = unit_records[:, :, 0:seq_length]
        e_records = np.array(unit_records_temp).flatten()

        num_neurons_DX = 1
        neurons_to_cover = sample_neurons(e_records, num_neurons_DX)

        # layers, fw_bw, batch, steps, rnn_size  ->  batch, layers, steps, fw_bw, rnn_size
        h_states_tensors = tf.transpose(h_states_all, perm=[2, 0, 3, 1, 4])
        h_states_tensors = tf.reshape(h_states_tensors, [batch, layers, steps, -1])
        h_states_tensors = tf.reduce_mean(h_states_tensors, axis=3)
        h_states_tensors = tf.reshape(h_states_tensors, [-1])

        for n_id in neurons_to_cover:
            neurons_tensor += h_states_tensors[n_id]
            neurons_list.append(h_states_tensors[n_id])

    return neurons_tensor, neurons_list


# random strategy to select states to boost
def neuron_selection_random(num_neurons, guided_coverage, h_states_all, c_states_all, seq_length):
    neurons_tensor = 0
    neurons_list = []
    print("neuron selection")
    if guided_coverage == "hidden_state":
        if len(hidden_state_records) <= 0:  # no records yet
            return h_states_all[0][0][0][0]

        hidden_state_temp = hidden_state_records[:, 0:seq_length, :]
        h_records = np.array(hidden_state_temp).flatten()
        neurons_to_cover = sample_neurons(h_records, num_neurons)
        h_states_tensors = tf.reshape(h_states_all, [-1])  # flatten the tensor
        for nid in neurons_to_cover:
            neurons_tensor += h_states_tensors[nid]
            neurons_list.append(h_states_tensors[nid])

    elif guided_coverage == "cell_state":
        if len(cell_state_records) <= 0:
            return c_states_all[0][0][0][0]

        c_lower_records = np.array(cell_state_records[:, 0:seq_length, :, 0]).flatten()
        c_higher_records = np.array(cell_state_records[:, 0:seq_length, :, len(tanh_sections) - 1]).flatten()

        # randomly sample uncovered neurons to cover
        lower_neurons = num_neurons // 2
        lower_neurons_to_cover = sample_neurons(c_lower_records, lower_neurons)

        higher_neurons = num_neurons - lower_neurons
        higher_neurons_to_cover = sample_neurons(c_higher_records, higher_neurons)

        c_states_tensors = tf.reshape(c_states_all, [-1])
        for ln_id in lower_neurons_to_cover:  # decrease the value of lower_neurons
            neurons_tensor -= c_states_tensors[ln_id]
            neurons_list.append(c_states_tensors[ln_id])
        for hn_id in higher_neurons_to_cover:  # increase the value of higher_neurons
            neurons_tensor += c_states_tensors[hn_id]
            neurons_list.append(c_states_tensors[hn_id])

    # elif guided_coverage == "input_gate":
    #     if len(input_gate_records) <= 0:
    #         return h_states_all[0][0][0][0], []
    #
    #     ig_lower_records = np.array(input_gate_records[0:seq_length, :, 0]).flatten()
    #     ig_higher_records = np.array(input_gate_records[0:seq_length, :, len(sigmoid_sections) - 1]).flatten()
    #
    #     lower_neurons = num_neurons / 2
    #     higher_neurons = num_neurons - lower_neurons
    #
    #     lower_neurons_to_cover = sample_neurons(ig_lower_records, lower_neurons)
    #     higher_neurons_to_cover = sample_neurons(ig_higher_records, higher_neurons)
    #
    #     e_num = np.array(input_gate_records).shape[1]
    #     h_states_tensors = tf.transpose(h_states_all, perm=[3, 0, 2, 1, 4])
    #     h_states_tensors = tf.reshape(h_states_tensors, [-1])
    #
    #     for index, ln_id in enumerate(lower_neurons_to_cover):
    #         tensor_id_value = ln_id - e_num
    #         neurons_tensor -= h_states_tensors[tensor_id_value]
    #         neurons_list.append(h_states_tensors[tensor_id_value])
    #     for index, hn_id in enumerate(higher_neurons_to_cover):
    #         tensor_id_value = hn_id - e_num
    #         neurons_tensor += h_states_tensors[tensor_id_value]
    #         neurons_list.append(h_states_tensors[tensor_id_value])

    # elif guided_coverage == "new_input_gate":
    #     if len(new_input_gate_records) <= 0:
    #         return h_states_all[0][0][0][0]
    #
    #     ni_lower_records = np.array(new_input_gate_records[0:seq_length, :, 0]).flatten()
    #     ni_higher_records = np.array(new_input_gate_records[0:seq_length, :, len(tanh_sections) - 1]).flatten()
    #
    #     lower_neurons = num_neurons / 2
    #     higher_neurons = num_neurons - lower_neurons
    #
    #     lower_neurons_to_cover = sample_neurons(ni_lower_records, lower_neurons)
    #     higher_neurons_to_cover = sample_neurons(ni_higher_records, higher_neurons)
    #
    #     e_num = np.array(new_input_gate_records).shape[1]
    #
    #     h_states_tensors = tf.transpose(h_states_all, perm=[3, 0, 2, 1, 4])
    #     h_states_tensors = tf.reshape(h_states_tensors, [-1])
    #     for index, ln_id in enumerate(lower_neurons_to_cover):
    #         tensor_id_value = ln_id - e_num + int(rnn_size) / 2
    #         neurons_tensor -= h_states_tensors[tensor_id_value]
    #     for index, hn_id in enumerate(higher_neurons_to_cover):
    #         tensor_id_value = hn_id - e_num + int(rnn_size) / 2
    #         neurons_tensor += h_states_tensors[tensor_id_value]

    elif guided_coverage == "forget_gate":
        if len(forget_gate_records) <= 0:
            return h_states_all[0][0][0][0]

        fg_lower_records = np.array(forget_gate_records[0:seq_length, :, 0]).flatten()
        fg_higher_records = np.array(forget_gate_records[0:seq_length, :, len(sigmoid_sections) - 1]).flatten()

        lower_neurons = num_neurons // 2
        higher_neurons = num_neurons - lower_neurons

        lower_neurons_to_cover = sample_neurons(fg_lower_records, lower_neurons)
        higher_neurons_to_cover = sample_neurons(fg_higher_records, higher_neurons)

        e_num = np.array(forget_gate_records).shape[1]
        h_states_tensors = tf.transpose(h_states_all, perm=[3, 0, 2, 1, 4])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])
        for index, ln_id in enumerate(lower_neurons_to_cover):
            tensor_id_value = ln_id - e_num
            if tensor_id_value < 0:
                continue
            neurons_tensor -= h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])
        for index, hn_id in enumerate(higher_neurons_to_cover):
            tensor_id_value = hn_id - e_num
            if tensor_id_value < 0:
                continue
            neurons_tensor += h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])

    elif guided_coverage == "output_gate":
        if len(output_gate_records) <= 0:
            return h_states_all[0][0][0][0]

        og_lower_records = np.array(output_gate_records[0:seq_length, :, 0]).flatten()
        og_higher_records = np.array(output_gate_records[0:seq_length, :, len(sigmoid_sections) - 1]).flatten()

        lower_neurons = num_neurons // 2
        higher_neurons = num_neurons - lower_neurons

        lower_neurons_to_cover = sample_neurons(og_lower_records, lower_neurons)
        higher_neurons_to_cover = sample_neurons(og_higher_records, higher_neurons)

        e_num = np.array(output_gate_records).shape[1]
        unit_size = (embedding_size + rnn_size) // 2
        h_states_tensors = tf.transpose(h_states_all, perm=[3, 0, 2, 1, 4])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])
        for index, ln_id in enumerate(lower_neurons_to_cover):
            tensor_id_value = ln_id - e_num + unit_size
            if tensor_id_value < 0:
                continue
            neurons_tensor -= h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])
        for index, hn_id in enumerate(higher_neurons_to_cover):
            tensor_id_value = hn_id - e_num + unit_size
            if tensor_id_value < 0:
                continue
            neurons_tensor += h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])

    # elif guided_coverage == "DX":
    #     if len(embedding_records) <= 0:
    #         return h_states_all[0][0][0][0]

    #     embedding_records_temp = embedding_records[:, :, 0:seq_length, :]
    #     e_records = np.array(embedding_records_temp).flatten()
    #     neurons_to_cover = sample_neurons(e_records, num_neurons)
    #     # layers, fw_bw, batch, steps, rnn_size  ->  batch, layers, steps, fw_bw, rnn_size
    #     h_states_tensors = tf.transpose(h_states_all, perm=[2, 0, 3, 1, 4])
    #     h_states_tensors = tf.reshape(h_states_tensors, [-1])

    #     for n_id in neurons_to_cover:
    #         neurons_tensor += h_states_tensors[n_id]

    return neurons_tensor


def sample_neurons(records, num_neurons):  # typeerror
    records_index = np.argsort(records)  # only 0 and 1 in h_records
    records_sorted = np.sort(records)

    # randomly sample uncovered neurons to cover
    records_inverse = np.subtract(max(records_sorted), records_sorted)
    if records_inverse[0] <= 0:  # all the neurons have been covered
        return [0]
    records_p_inverse = records_inverse / float(sum(records_inverse))

    uncovered_neurons = 1 / records_p_inverse[0]
    if uncovered_neurons < num_neurons:
        num_neurons = int(uncovered_neurons)

    neurons_to_cover = np.random.choice(records_index, num_neurons, replace=False, p=records_p_inverse)
    return neurons_to_cover


if __name__ == "__main__":
    pass



