"""
Coverage Criteria for the Testing of Language Modeling
include both ours and those of DeepXplore and DeepGauge
"""

import numpy as np
from scipy.special import expit
import random
import tensorflow as tf
import copy
import time

hidden_state_records = []
cell_state_records = []

input_gate_records = []
forget_gate_records = []
output_gate_records = []
new_input_gate_records = []

cell_tanh_sections = [-0.8, -0.2, 0.2, 0.8]  # according to the distribution of tanh function
# < the lower bound & > the upper bound are the two boundary sections
# sections for the records
sigmoid_sections_rec = [0.2, 0.4, 0.6, 0.8]
tanh_sections_rec = [-0.75, -0.2, 0.2, 0.75]

# sections for the coverage boosting
sigmoid_sections = [0.1, 0.4, 0.6, 0.9]  # according to the distribution of sigmoid function
tanh_sections = [-0.8, -0.2, 0.2, 0.8]  # according to the distribution of tanh function

threshold_hidden = 0.01
threshold_deepXplore = 0.5
embedding_records = []
unit_records = []


# ---------------------------------#
# Coverage Criteria of ours
def hidden_state_coverage(h_states_all_value):  # can't boost the coverage
    steps, layers, batch, embedding = np.array(h_states_all_value).shape
    h_all = h_states_all_value.flatten()
    h_all = np.reshape(h_all, [-1, embedding])
    top_num = int(embedding * threshold_hidden)

    global hidden_state_records
    if hidden_state_records == []:
        hidden_state_records = np.zeros_like(h_all)

    top_neurons = 0
    for e, e_array in enumerate(h_all):
        # neuron_id = np.argmax(e_array)
        neuron_ids = np.array(e_array).argsort()[-top_num:]

        for neuron_id in neuron_ids:
            hidden_state_records[e][neuron_id] = 1  # update
        for n, neuron in enumerate(e_array):
            if hidden_state_records[e][n] == 1:  # record
                top_neurons += 1

    hidden_state_c = float(top_neurons) / (steps * layers * batch * embedding)
    return hidden_state_c


def cell_state_coverage(c_states_all_value):  # use the tanh sections
    steps, layers, batch, embedding = np.array(c_states_all_value).shape
    sections = len(cell_tanh_sections) + 1

    c_all = c_states_all_value.flatten()
    c_all = np.reshape(c_all, [-1, embedding])

    global cell_state_records
    if cell_state_records == []:
        cell_state_records = np.zeros([steps * layers * batch, embedding, sections])

    cell_state_c = np.zeros(sections, dtype=float)
    for e, e_array in enumerate(c_all):
        e_array = np.tanh(e_array)
        for n, neuron in enumerate(e_array):
            # update
            isset = False
            for sec, sec_value in enumerate(cell_tanh_sections):
                if neuron < sec_value:
                    cell_state_records[e][n][sec] = 1
                    isset = True
                    break
            if not isset:
                cell_state_records[e][n][sections - 1] = 1

            # record
            for sec in range(sections):
                if cell_state_records[e][n][sec] == 1:
                    cell_state_c[sec] += 1

    cell_state_c /= layers * steps * batch * embedding
    return cell_state_c


def gate_coverage(gates_all):
    layers, batch, steps, embedding = np.array(gates_all).shape  # (layer+1, batch, step+1, embedding)
    gates_all = gates_all.swapaxes(1, 2)  # layers, steps, batch, embedding

    # store the values of each time
    input_gate_all = np.zeros([layers - 1, steps - 1, batch, embedding/2])
    forget_gate_all = np.zeros_like(input_gate_all)
    output_gate_all = np.zeros_like(input_gate_all)
    new_input_gate_all = np.zeros_like(input_gate_all)

    # construct the experiment data for each gate
    for l in range(1, layers):  # the loop starts from layer 1, as the layer 0 here stores the input
        for s in range(1, steps):
            # according to the computation of each lstm cell, concatenate two embeddings and split to four parts
            concat = np.concatenate((gates_all[l-1][s], gates_all[l][s-1]), axis=1)
            input_gates, new_input_gates, forget_gates, output_gates = np.split(concat, 4, axis=1)

            l_index = l-1  # the record index starts from 0
            s_index = s-1
            # Also corresponds to the activation functions each gate uses
            input_gate_all[l_index][s_index] = expit(input_gates)  # expit: sigmoid
            new_input_gate_all[l_index][s_index] = np.tanh(new_input_gates)
            forget_gate_all[l_index][s_index] = expit(forget_gates)
            output_gate_all[l_index][s_index] = expit(output_gates)

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
        e_num = (layers - 1) * (steps - 1) * batch * embedding / 2
        input_gate_records = np.zeros([e_num, sections_num[0]])
        forget_gate_records = np.zeros([e_num, sections_num[1]])
        new_input_gate_records = np.zeros([e_num, sections_num[2]])
        output_gate_records = np.zeros([e_num, sections_num[3]])

    input_gate_c = each_gate_coverage(input_gate_all, activation_f[0], input_gate_records)
    new_input_c = each_gate_coverage(new_input_gate_all, activation_f[1], new_input_gate_records)
    forget_gate_c = each_gate_coverage(forget_gate_all, activation_f[2], forget_gate_records)
    output_gate_c = each_gate_coverage(output_gate_all, activation_f[3], output_gate_records)

    return input_gate_c, new_input_c, forget_gate_c, output_gate_c


def each_gate_coverage(gate_array, activation_f, records):
    layers, steps, batch, embedding = gate_array.shape
    sections_num = records.shape[-1]
    if activation_f == "sigmoid":
        sections = sigmoid_sections_rec
    else:
        sections = tanh_sections_rec

    gate_c = np.zeros(sections_num, dtype=float)

    gate_all = gate_array.flatten()
    for e, e_value in enumerate(gate_all):
        # update
        isset = False
        for sec, sec_value in enumerate(sections):
            if e_value < sec_value:
                records[e][sec] = 1
                isset = True
                break
        if not isset:
            records[e][sections_num - 1] = 1

        # record
        for sec in range(sections_num):
            if records[e][sec] == 1:
                gate_c[sec] += 1

    gate_c /= layers * steps * batch * embedding
    return gate_c


# ---------------------------------#
# coverage criteria like DeepXplore
def embedding_neuron_coverage(h_states_all_value):
    batch, layers, steps, embedding = np.array(h_states_all_value).shape
    h_all = h_states_all_value.flatten()
    h_all = np.reshape(h_all, [batch, layers, steps * embedding])

    global embedding_records
    if len(embedding_records) <= 0:
        embedding_records = np.zeros_like(h_all)

    neurons_covered = 0
    for b, batch_array in enumerate(h_all):
        for l, layer_array in enumerate(batch_array):
            scaled_layer = scale(np.array(layer_array))
            for e, embedding_value in enumerate(scaled_layer):
                if embedding_value >= threshold_deepXplore:
                    embedding_records[b][l][e] = 1  # covered
                if embedding_records[b][l][e] > 0:
                    neurons_covered += 1

    all_neurons = batch * layers * steps * embedding
    coverage = float(neurons_covered) / all_neurons
    return coverage


# not used for guiding
def unit_neuron_coverage(h_states_all_value):
    batch, layers, steps, embedding = np.array(h_states_all_value).shape
    h_mean = np.mean(h_states_all_value, axis=3)

    global unit_records
    if unit_records == []:
        unit_records = np.zeros_like(h_mean)

    units_covered = 0
    for b, batch_array in enumerate(h_mean):
        for l, layer_array in enumerate(batch_array):
            scaled_layer = scale(np.array(layer_array))
            for s, step_value in enumerate(scaled_layer):
                if step_value >= threshold_deepXplore:  # update the coverage
                    unit_records[b][l][s] = 1  # covered
                if unit_records[b][l][s] > 0:  # to compute the coverage
                    units_covered += 1

    all_units = batch * layers * steps
    coverage = float(units_covered) / all_units
    return coverage


# necessary functions
def scale(output, max=1, min=0):
    std = (output - output.min()) / (output.max() - output.min())
    scaled = std * (max - min) + min
    return scaled


# ---------------------------------#
# coverage criteria like DeepGauge
def neuron_boundary_coverage(h_states_all_value):
    pass


def neuron_selection_random(num_neurons, guided_coverage, h_states_all, c_states_all, inputs):
    neurons_tensor = 0
    neurons_list = []
    if guided_coverage == "hidden_state":
        if len(hidden_state_records) <= 0:  # no records yet
            return h_states_all[0][0][0][0]

        h_records = np.array(hidden_state_records).flatten()
        neurons_to_cover = sample_neurons(h_records, num_neurons)
        h_states_tensors = tf.reshape(h_states_all, [-1])  # flatten the tensor
        for nid in neurons_to_cover:
            neurons_tensor += h_states_tensors[nid]
            neurons_list.append(h_states_tensors[nid])

    elif guided_coverage == "cell_state":
        if len(cell_state_records) <= 0:
            return c_states_all[0][0][0][0]

        c_lower_records = np.array(cell_state_records[:, :, 0]).flatten()
        c_higher_records = np.array(cell_state_records[:, :, len(tanh_sections) - 1]).flatten()

        # randomly sample uncovered neurons to cover
        lower_neurons = num_neurons / 2
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

    elif guided_coverage == "input_gate":
        if len(input_gate_records) <= 0:
            return h_states_all[0][0][0][0], []

        ig_lower_records = np.array(input_gate_records[:, 0]).flatten()
        ig_higher_records = np.array(input_gate_records[:, len(sigmoid_sections) - 1]).flatten()

        lower_neurons = num_neurons / 2
        higher_neurons = num_neurons - lower_neurons

        lower_neurons_to_cover = sample_neurons(ig_lower_records, lower_neurons)
        higher_neurons_to_cover = sample_neurons(ig_higher_records, higher_neurons)

        steps, layers, batch, embedding = h_states_all.get_shape()
        h_states_tensors = tf.transpose(h_states_all, perm=[1, 0, 2, 3])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])

        for index, ln_id in enumerate(lower_neurons_to_cover):
            tensor_id_value = ln_id - int(layers) * int(batch) * int(embedding)
            neurons_tensor -= h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])
        for index, hn_id in enumerate(higher_neurons_to_cover):
            tensor_id_value = hn_id - int(layers) * int(batch) * int(embedding)
            neurons_tensor += h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])

    elif guided_coverage == "new_input_gate":
        if len(new_input_gate_records) <= 0:
            return h_states_all[0][0][0][0]

        ni_lower_records = np.array(new_input_gate_records[:, 0]).flatten()
        ni_higher_records = np.array(new_input_gate_records[:, len(tanh_sections) - 1]).flatten()

        lower_neurons = num_neurons / 2
        higher_neurons = num_neurons - lower_neurons

        lower_neurons_to_cover = sample_neurons(ni_lower_records, lower_neurons)
        higher_neurons_to_cover = sample_neurons(ni_higher_records, higher_neurons)

        steps, layers, batch, embedding = h_states_all.get_shape()
        h_states_tensors = tf.transpose(h_states_all, perm=[1, 0, 2, 3])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])
        for index, ln_id in enumerate(lower_neurons_to_cover):
            tensor_id_value = ln_id - int(layers) * int(batch) * int(embedding) + int(embedding) / 2
            neurons_tensor -= h_states_tensors[tensor_id_value]
        for index, hn_id in enumerate(higher_neurons_to_cover):
            tensor_id_value = hn_id - int(layers) * int(batch) * int(embedding) + int(embedding) / 2
            neurons_tensor += h_states_tensors[tensor_id_value]

    elif guided_coverage == "forget_gate":
        if len(forget_gate_records) <= 0:
            return h_states_all[0][0][0][0]

        fg_lower_records = np.array(forget_gate_records[:, 0]).flatten()
        fg_higher_records = np.array(forget_gate_records[:, len(sigmoid_sections) - 1]).flatten()

        lower_neurons = num_neurons / 2
        higher_neurons = num_neurons - lower_neurons

        lower_neurons_to_cover = sample_neurons(fg_lower_records, lower_neurons)
        higher_neurons_to_cover = sample_neurons(fg_higher_records, higher_neurons)

        steps, layers, batch, embedding = h_states_all.get_shape()
        h_states_tensors = tf.transpose(h_states_all, perm=[1, 0, 2, 3])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])
        for index, ln_id in enumerate(lower_neurons_to_cover):
            tensor_id_value = ln_id - int(batch) * int(embedding)
            neurons_tensor -= h_states_tensors[tensor_id_value]
        for index, hn_id in enumerate(higher_neurons_to_cover):
            tensor_id_value = hn_id - int(batch) * int(embedding)
            neurons_tensor += h_states_tensors[tensor_id_value]

    elif guided_coverage == "output_gate":
        if len(output_gate_records) <= 0:
            return h_states_all[0][0][0][0], []

        og_lower_records = np.array(output_gate_records[:, 0]).flatten()
        og_higher_records = np.array(output_gate_records[:, len(sigmoid_sections) - 1]).flatten()

        lower_neurons = num_neurons / 2
        higher_neurons = num_neurons - lower_neurons

        lower_neurons_to_cover = sample_neurons(og_lower_records, lower_neurons)
        higher_neurons_to_cover = sample_neurons(og_higher_records, higher_neurons)

        steps, layers, batch, embedding = h_states_all.get_shape()
        h_states_tensors = tf.transpose(h_states_all, perm=[1, 0, 2, 3])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])
        for index, ln_id in enumerate(lower_neurons_to_cover):
            tensor_id_value = ln_id - int(batch) * int(embedding) + int(embedding) / 2
            neurons_tensor -= h_states_tensors[tensor_id_value]
        for index, hn_id in enumerate(higher_neurons_to_cover):
            tensor_id_value = hn_id - int(batch) * int(embedding) + int(embedding) / 2
            neurons_tensor += h_states_tensors[tensor_id_value]

    elif guided_coverage == "DX":  # not verified
        if len(embedding_records) <= 0:
            return h_states_all[0][0][0][0]

        e_records = np.array(embedding_records).flatten()
        neurons_to_cover = sample_neurons(e_records, num_neurons)

        h_states_tensors = tf.transpose(h_states_all, perm=[2, 1, 0, 3])
        h_states_tensors = tf.reshape(h_states_tensors, [-1])

        for n_id in neurons_to_cover:
            neurons_tensor += h_states_tensors[n_id]

    return neurons_tensor  #, neurons_list


def neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all, h_states_all_value,
                     c_states_all_value):
    neurons_tensor = 0
    neurons_list = []
    if guided_coverage == "hidden_state":
        steps, layers, batch, embedding = np.array(h_states_all_value).shape
        top_neurons = int(steps * layers * batch * embedding * threshold_hidden)
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
            # neurons_list.append(h_states_tensors[nid])

    elif guided_coverage == "cell_state":
        lower_neurons = num_neurons / 2
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
            # neurons_list.append(c_states_tensors[ln_id])
        for hn_id in higher_neuron_ids:  # increase the value of higher_neurons
            neurons_tensor += c_states_tensors[hn_id]
            # neurons_list.append(c_states_tensors[hn_id])

    elif guided_coverage == "new_input_gate":
        lower_neurons = num_neurons / 2
        higher_neurons = num_neurons - lower_neurons
        h_all = h_states_all_value.flatten()

        new_input_tanh = np.tanh(h_all)
        e_array_sorted = sorted(new_input_tanh)
        e_array_sorted_reverse = sorted(new_input_tanh, reverse=True)

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
            higher_neuron_ids = np.array(new_input_tanh).argsort()[-(high_start + higher_neurons):-high_start]
        else:
            higher_neuron_ids = np.array(new_input_tanh).argsort()[-higher_neurons:]
        lower_neuron_ids = np.array(new_input_tanh).argsort()[low_start:(low_start + lower_neurons)]

        h_states_tensors = tf.reshape(h_states_all, [-1])
        for index, ln_id in enumerate(lower_neuron_ids):
            tensor_id_value = ln_id
            neurons_tensor -= h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])
        for index, hn_id in enumerate(higher_neuron_ids):
            tensor_id_value = hn_id
            neurons_tensor += h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])

    elif guided_coverage in ["input_gate", "forget_gate", "output_gate"]:
        lower_neurons = num_neurons / 2
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
            # neurons_list.append(h_states_tensors[tensor_id_value])
        for index, hn_id in enumerate(higher_neuron_ids):
            tensor_id_value = hn_id
            neurons_tensor += h_states_tensors[tensor_id_value]
            # neurons_list.append(h_states_tensors[tensor_id_value])
    elif guided_coverage == "DX":
        if len(unit_records) <= 0:
            return h_states_all[0][0][0][0]

        unit_records_temp = unit_records
        e_records = np.array(unit_records_temp).flatten()

        num_neurons_DX = 1
        neurons_to_cover = sample_neurons(e_records, num_neurons_DX)

        h_states_tensors = tf.transpose(h_states_all, perm=[2, 1, 0, 3])
        h_states_tensors = tf.reduce_mean(h_states_tensors, axis=3)
        h_states_tensors = tf.reshape(h_states_tensors, [-1])

        for n_id in neurons_to_cover:
            neurons_tensor += h_states_tensors[n_id]
            neurons_list.append(h_states_tensors[n_id])

    # elif guided_coverage == "DX":  # before
    #     h_all = h_states_all_value.flatten()
    #     h_array_sorted = sorted(h_all)
    #     # print(h_all.argsort())
    #     start = 0
    #     h_array_sorted = scale(np.array(h_array_sorted))
    #     for i, e_value in enumerate(h_array_sorted):
    #         if e_value > threshold_deepXplore * 0.6:
    #             start = i
    #             break
    #     neuron_ids = np.array(h_all).argsort()[start:(start + num_neurons)]  # ....
    #
    #     h_states_tensors = tf.reshape(h_states_all, [-1])
    #     for nid in neuron_ids:
    #         neurons_tensor += h_states_tensors[nid]
    #         # neurons_list.append(h_states_tensors[nid])
    #
    # elif guided_coverage == "DX":  # more neurons and near-activation selection
    #     h_array = h_states_all_value.swapaxes(0, 2)  # batch, layer, step, embedding
    #     h_mean = np.mean(h_array, axis=3)
    #     print("h_mean")
    #     print(h_mean.shape)
    #     h_all = h_mean.flatten()
    #
    #     h_array_sorted = sorted(h_all)
    #
    #     start = 0
    #     h_array_sorted = scale(np.array(h_array_sorted))
    #     for i, e_value in enumerate(h_array_sorted):
    #         if e_value > threshold_deepXplore * 0.6:
    #             start = i
    #             break
    #
    #     neuron_ids = np.array(h_all).argsort()[start:(start + num_neurons)]
    #
    #     h_states_tensors = tf.transpose(h_states_all, perm=[2, 1, 0, 3])
    #     h_states_tensors = tf.reduce_mean(h_states_tensors, axis=3)
    #     print("h states tensors")
    #     print(h_states_tensors)
    #     h_states_tensors = tf.reshape(h_states_tensors, [-1])
    #
    #     for nid in neuron_ids:
    #         neurons_tensor += h_states_tensors[nid]
    return neurons_tensor


def sample_neurons(records, num_neurons):
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


# def sample_section_boundary_neurons(records, num_neurons):
#     lower_records = np.array(records[:, 0]).flatten()
#     higher_records = np.array(records[:, len(tanh_sections) - 1]).flatten()  # the ac_f
#
#     lower_neurons = num_neurons / 2
#     higher_neurons = num_neurons - lower_neurons
#
#     lower_neurons_to_cover = sample_neurons(lower_records, lower_neurons)
#     higher_neurons_to_cover = sample_neurons(higher_records, higher_neurons)
#     return lower_neurons_to_cover, higher_neurons_to_cover


if __name__ == "__main__":
    pass
