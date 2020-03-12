from SpellChecker_wrapped import *
from coverage_sp import *
import random
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
import Levenshtein
import csv
import os
import collections

_FLOATX = 'float32'

flags = tf.flags
flags.DEFINE_integer('topk', 3, 'topk')
flags.DEFINE_string('guided_coverage',
                    "cell_state",
                    'guided_coverage, could be "hidden_state","cell_state","DX", "forget_gate","output_gate"')
flags.DEFINE_integer('times_nn', 5, 'num_neurons = len(text) / times_nn')
flags.DEFINE_integer('objective', 2, '0 for objective1, 1 for objective2, 2 for objective1 + objective2')
flags.DEFINE_string('objective1', "state_diff", 'objective1 could be "diff", "cost", "state_diff"')
# flags.DEFINE_integer('times_ns', 3, 'num_samples = model.batch_size * model.num_steps * 3')
# flags.DEFINE_integer('lr_l', 0, "the lower bound of the learning rate")
# flags.DEFINE_integer('lr_u', 70, 'the upper bound of the learning rate')
flags.DEFINE_string('exp', "adv", 'exp could be "baseline" and "adv"')

FLAGS = flags.FLAGS


def run_exp(sess, model, texts, targets, csv_filename):
    max_sen_length = max_length  # defined in the model file
    success_num_all = 0
    tested_texts_num = 0
    generation_num = 0
    generation_guided_num = 0
    csv_dict = []  # save the value that need to be saved to a csv file

    for text_id in range(len(texts)):  # collect some data and run together
        step_dict = collections.OrderedDict()
        step_dict['text_id'] = text_id

        text = texts[text_id]
        step_dict['text'] = text
        print("  Input Words: ", text)
        text_str = text
        text = text_to_ints(text)

        target = targets[text_id]
        target_str = target
        step_dict['target'] = target_str
        target_chars = characters(target_str)
        target = text_to_ints(target)

        # originally predict
        c_states_all, h_states_all = extract_c_h_tensors(model.layers_enc_outputs)
        fetches = [model.cost, model.predictions, model.input_embedding, model.input_reverse,
                   model.inputs_all_embeddings, c_states_all, h_states_all, model.rnn_outputs]

        # Multiply by batch_size to match the model's input parameters
        cost, answer_logits, input_embedding, input_reverse, inputs_all_embeddings, c_states_all_value, \
        h_states_all_value, rnn_outputs = sess.run(fetches,
                               {model.inputs: [text] * batch_size,
                                model.inputs_length: [len(text)] * batch_size,
                                model.targets: [target] * batch_size,
                                model.targets_length: ([len(target)]) * batch_size,
                                model.keep_prob: [1.0]})

        # print("cost: ", cost)
        predicted_seq = output_seq(prefix="  Original Response Words: ", text=answer_logits)
        ori_response_chars = [characters(predicted_seq)]
        ori_pairs = correct_pairs(text_str, target_str, predicted_seq)
        if len(ori_pairs) <= 0:
            continue
        else:
            tested_texts_num += 1

        ori_WER = Levenshtein.distance(predicted_seq, target_str)
        ori_bleu = sentence_bleu(ori_response_chars, target_chars)
        # print("ori_cost: ", cost)
        # print("ori_bleu: ", ori_bleu)
        # print("ori_wer: ", ori_WER)
        step_dict['cost_ori'] = cost
        step_dict['bleu_ori'] = ori_bleu
        step_dict['WER_ori'] = ori_WER

        inputs = np.append(np.expand_dims(input_embedding, axis=0), np.expand_dims(input_reverse, axis=0), axis=0)

        guided_coverage = FLAGS.guided_coverage
        # update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage,
        #                 max_sen_length, step_dict, label="ori")

        word_to_modify = ori_pairs[0][0]
        start_index = text_str.find(word_to_modify)
        end_index = start_index + len(word_to_modify) - 1
        # first_index = 0
        while True:
            step_index = random.randint(start_index, end_index)
            # step_index = start_index
            first_index = step_index
            # step_index = random.randint(1, len(text) - 1)
            origin_vocab = int_to_vocab[text[step_index]]
            if origin_vocab != ' ':
                break

        if FLAGS.exp == "baseline":
            replaced_sentence = generate_replaced_sequences(text, step_index)
            cost_base, bleu_base, WER_base, h_states_base, c_states_base, inputs_base, suc_base = predict(model, sess,
                                                                                                          replaced_sentence,
                                                                                                          target,
                                                                                                          step_index,
                                                                                                          ori_pairs,
                                                                                                          h_states_all,
                                                                                                          c_states_all)
            step_dict['cost_base'] = cost_base
            step_dict['bleu_base'] = bleu_base
            step_dict['WER_base'] = WER_base

            step_dict['cost_inc'] = cost_base - cost
            step_dict['bleu_dec'] = ori_bleu - bleu_base
            step_dict['WER_inc'] = WER_base - ori_WER

            if suc_base:
                success_num_all += 1
            update_coverage(h_states_base, c_states_base, inputs_base, guided_coverage,
                            max_sen_length, step_dict, label="base")  # adv, should be base
            csv_dict.append(step_dict)
            generation_num += 1
        else:
            while True:
                adv_start = time.clock()
                if FLAGS.objective == 0:
                    objective1 = get_objective1(model, step_index, h_states_all, c_states_all, model.rnn_outputs,
                                                rnn_outputs)
                    objective = objective1
                elif FLAGS.objective == 1:
                    num_neurons = len(text) / FLAGS.times_nn
                    print("num_neurons: ", num_neurons)
                    objective2, neurons_list = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all,
                                                                h_states_all_value,
                                                                c_states_all_value, len(text))
                    print("neurons_list: ", len(neurons_list))
                    objective = objective2
                elif FLAGS.objective == 2:
                    objective1 = get_objective1(model, step_index, h_states_all, c_states_all, model.rnn_outputs,
                                                rnn_outputs)
                    num_neurons = len(text) / FLAGS.times_nn
                    objective2, neurons_list = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all,
                                                                h_states_all_value,
                                                                c_states_all_value, len(text))
                    objective = objective1 + objective2
                else:
                    objective = 0
                    print('Invalid input.  0 for objective1, 1 for objective2, 2 for objective1 + objective2')
                print("getting grads: ")
                grads = tf.gradients(objective, model.input_embedding)
                grads_value = sess.run(grads,
                                       {model.inputs: [text] * batch_size,
                                        model.inputs_length: [len(text)] * batch_size,
                                        model.targets: [target] * batch_size,
                                        model.targets_length: ([len(target)]) * batch_size,
                                        model.keep_prob: [1.0]})
                # print("value_list")
                # print(value_list)
                adv_sentence = generate_adv_sequences(text, input_embedding, grads_value, inputs_all_embeddings,
                                                      step_index, step_dict)

                adv_end = time.clock()
                is_generate = is_generate_adv(text, adv_sentence)
                print("is generate: ", is_generate)
                # step_dict['is_generate'] = is_generate

                duration = adv_end - adv_start
                step_dict['adv_time'] = duration

                # predict after modification
                if is_generate:
                    cost_adv, bleu_adv, WER_adv, h_states_all_adv, c_states_all_adv, inputs_adv, suc_adv = predict(model, sess,
                                                                                                            adv_sentence,
                                                                                                            target,
                                                                                                            step_index,
                                                                                                            ori_pairs,
                                                                                                            h_states_all,
                                                                                                            c_states_all)
                    step_dict['cost_adv'] = cost_adv
                    step_dict['bleu_adv'] = bleu_adv
                    step_dict['WER_adv'] = WER_adv
                    step_dict['cost_inc'] = cost_adv - cost
                    step_dict['bleu_dec'] = ori_bleu - bleu_adv
                    step_dict['WER_inc'] = WER_adv - ori_WER

                    if suc_adv:
                        success_num_all += 1
                    update_coverage(h_states_all_adv, c_states_all_adv, inputs_adv, guided_coverage,
                                    max_sen_length, step_dict, label="adv")
                    generation_num += 1
                    generation_guided_num += 1
                    break
                else:
                    new_step_index = step_index + 1
                    if new_step_index >= end_index:
                        new_step_index = start_index
                    print("new_step_index: ", new_step_index)
                    print("first index:", first_index)
                    if new_step_index != first_index:
                        step_index = new_step_index
                        continue
                    else:  # random
                        replaced_sentence = generate_replaced_sequences(text, step_index)
                        cost_base, bleu_base, WER_base, h_states_base, c_states_base, inputs_base, suc_base = predict(
                            model, sess,
                            replaced_sentence,
                            target,
                            step_index,
                            ori_pairs,
                            h_states_all,
                            c_states_all)
                        step_dict['cost_adv'] = cost_base
                        step_dict['bleu_adv'] = bleu_base
                        step_dict['WER_adv'] = WER_base

                        step_dict['cost_inc'] = cost_base - cost
                        step_dict['bleu_dec'] = ori_bleu - bleu_base
                        step_dict['WER_inc'] = WER_base - ori_WER

                        if suc_base:
                            success_num_all += 1
                        update_coverage(h_states_base, c_states_base, inputs_base, guided_coverage,
                                        max_sen_length, step_dict, label="adv")
                        generation_num += 1
                        break

                    # cost_adv, bleu_adv, WER_adv = 0, 0, 0
                    # step_dict['cost_adv'] = cost_adv
                    # step_dict['bleu_adv'] = bleu_adv
                    # step_dict['WER_adv'] = WER_adv
                    # step_dict['cost_inc'] = 0
                    # step_dict['bleu_dec'] = 0
                    # step_dict['WER_inc'] = 0
                    # # step_dict['suc_adv'] = False
                    # update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage,
                    #                 max_sen_length, step_dict, label="adv")  # as the original
            csv_dict.append(step_dict)

    if generation_num <= 0:
        success_rate = 0
    else:
        success_rate = float(success_num_all) / generation_num
    print("success_rate: ", success_rate)

    save_csv(csv_dict, csv_filename)
    print('save success')
    return success_rate, float(generation_num) / tested_texts_num, float(generation_guided_num) / generation_num


def run_ori(sess, model, texts, targets, csv_filename):
    max_sen_length = max_length  # defined in the model file
    success_num_all = 0
    tested_texts_num = 0
    csv_dict = []  # save the value that need to be saved to a csv file

    for text_id in range(len(texts)):  # collect some data and run together
        step_dict = collections.OrderedDict()
        step_dict['text_id'] = text_id

        text = texts[text_id]
        step_dict['text'] = text
        # print("  Input Words: ", text)
        text_str = text
        text = text_to_ints(text)

        target = targets[text_id]
        target_str = target
        step_dict['target'] = target_str
        # target_chars = characters(target_str)
        target = text_to_ints(target)

        # originally predict
        c_states_all, h_states_all = extract_c_h_tensors(model.layers_enc_outputs)
        fetches = [model.predictions, model.input_embedding, model.input_reverse, c_states_all, h_states_all]

        # Multiply by batch_size to match the model's input parameters
        answer_logits, input_embedding, input_reverse, c_states_all_value, h_states_all_value = sess.run(fetches,
                                                   {model.inputs: [text] * batch_size,
                                                    model.inputs_length: [len(text)] * batch_size,
                                                    model.targets: [target] * batch_size,
                                                    model.targets_length: ([len(target)]) * batch_size,
                                                    model.keep_prob: [1.0]})

        # print("cost: ", cost)
        predicted_seq = output_seq(prefix="  Original Response Words: ", text=answer_logits)
        # ori_response_chars = [characters(predicted_seq)]
        ori_pairs = correct_pairs(text_str, target_str, predicted_seq)
        if len(ori_pairs) <= 0:
            continue
        # else:
        #     tested_texts_num += 1
        #
        # ori_WER = Levenshtein.distance(predicted_seq, target_str)
        # ori_bleu = sentence_bleu(ori_response_chars, target_chars)
        # # print("ori_cost: ", cost)
        # # print("ori_bleu: ", ori_bleu)
        # # print("ori_wer: ", ori_WER)
        # step_dict['cost_ori'] = cost
        # step_dict['bleu_ori'] = ori_bleu
        # step_dict['WER_ori'] = ori_WER

        inputs = np.append(np.expand_dims(input_embedding, axis=0), np.expand_dims(input_reverse, axis=0), axis=0)

        guided_coverage = FLAGS.guided_coverage
        update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage,
                        max_sen_length, step_dict, label="ori")
        csv_dict.append(step_dict)

    save_csv(csv_dict, csv_filename)
    print('save success')


def generate_replaced_sequences(text, step_index):
    replaced_sequence = np.copy(text)
    while True:
        char_id = random.randint(0, len(int_to_vocab) - 1)
        if int_to_vocab[char_id] == '<PAD>' or \
                        int_to_vocab[char_id] == '<EOS>' or \
                        int_to_vocab[char_id] == '<GO>' or \
                        int_to_vocab[char_id] == ' ':
            continue
        else:
            break
    # char_id = random.randint(0, len(int_to_vocab) - 1)
    replaced_sequence[step_index] = char_id
    return replaced_sequence


def generate_adv_sequences(text, input_value, grads_value, embedding, step_index, step_dict):
    # text, input_value, grads_value, embedding):
    adv_sequence = np.copy(text)
    input_value = input_value[0]  # [::-1] should be verified
    grads_value = grads_value[0][0]

    step_dict['perturbation'] = 0

    origin_id = text[step_index]

    max_scale = 200
    pert = grads_value[step_index]
    # pert = grads_value[step_index] / np.linalg.norm(grads_value[step_index])
    input_emb = input_value[step_index]

    for p_scale in range(1, max_scale, 2):
        pert_scale = pert * p_scale
        temp_emb = input_emb + pert_scale

        dist = []
        for emb in embedding:
            dist.append(np.linalg.norm(emb - temp_emb))

        min_id = int(np.argmin(dist))

        if min_id >= len(int_to_vocab):
            continue
        if int_to_vocab[min_id] == '<PAD>' or \
                        int_to_vocab[min_id] == '<EOS>' or \
                        int_to_vocab[min_id] == '<GO>' or \
                        int_to_vocab[min_id] == ' ':
            continue
        if origin_id != min_id:
            # print("p_scale, ", p_scale)
            step_dict['perturbation'] = np.linalg.norm(pert_scale)
            # print(step_dict['perturbation'])
            # print(step_dict['perturbation'] /np.linalg.norm(input_emb))
            # print("input_emb, ", input_emb)
            # print("pert_scale, ", pert_scale)
            # print("temp_emb, ", temp_emb)
            # cur_vocab = int_to_vocab[min_id]
            adv_sequence[step_index] = min_id
            break
    return adv_sequence


def save_csv(dict, filename):
    if os.path.exists('./experiment_result_boost'):
        pass
    else:
        os.makedirs('./experiment_result_boost')
    filename = './experiment_result_boost/'+filename
    with open(filename, 'wb') as f:
        w = csv.writer(f)
        fields_names = dict[0].keys()
        w.writerow(fields_names)
        for row in dict:
            w.writerow(row.values())


def get_objective1(model, step, h_states_all, c_states_all, output_tensor, rnn_outputs):
    objective1 = 0
    if FLAGS.objective1 == "diff":
        predict_length = np.array(rnn_outputs).shape[1]
        if step >= predict_length:
            rnn_outputs = rnn_outputs[0][predict_length - 1]
        else:
            rnn_outputs = rnn_outputs[0][step]  # why 0 because repeat
        # print("softmax_array")
        # print(rnn_outputs)
        top_idx = np.argsort(rnn_outputs)[::-1]
        # print("top_idx")
        # print(top_idx)
        for i in range(1, FLAGS.topk):
            objective1 += mean(output_tensor[0][step][top_idx[i]])

        objective1 -= mean(output_tensor[0][step][top_idx[0]])

    elif FLAGS.objective1 == "cost":
        objective1 = model.cost
    elif FLAGS.objective1 == "state_diff":
        # layers, fw_bw, batch, steps, rnn_size -> steps, layers, fw_bw, batch, rnn_size
        h_states_all = tf.transpose(h_states_all, perm=[3, 1, 2, 0, 4])
        c_states_all = tf.transpose(c_states_all, perm=[3, 1, 2, 0, 4])
        objective1 = h_states_all[step-1] + c_states_all[step] - h_states_all[step]  # state diff
    else:
        print("Objective1 not configured.")
    return objective1


def is_generate_adv(x, y):
    result = (x == y)
    if False in result:
        return True
    else:
        return False


def mean(x, axis=None, keepdims=False):
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    return tf.reduce_mean(x, axis, keepdims)


def update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage, max_sen_length, step_dict, label):
    # compute the state coverage
    if guided_coverage == "hidden_state":
        if label == "nan":
            step_dict['hidden_state_c_adv'] = np.nan
        else:
            hidden_state_c = hidden_state_coverage(h_states_all_value, max_sen_length)
            if label == "ori":
                step_dict['hidden_state_c_ori'] = str(hidden_state_c)
                # print("hidden_state_c_0:", hidden_state_c)
            elif label == "base":
                step_dict['hidden_state_c_base'] = str(hidden_state_c)
            elif label == "adv":
                step_dict['hidden_state_c_adv'] = str(hidden_state_c)
    elif guided_coverage == "cell_state":
        if label == "nan":
            save_coverage('cell_state_c_adv', [np.nan, np.nan, np.nan, np.nan, np.nan], step_dict)
        else:
            cell_state_c = cell_state_coverage(c_states_all_value, max_sen_length)
            if label == "ori":
                print("cell_state_c_ori:", cell_state_c)
                save_coverage('cell_state_c_ori',cell_state_c,step_dict)
            elif label == "base":
                save_coverage('cell_state_c_base', cell_state_c,step_dict)
            elif label == "adv":
                save_coverage('cell_state_c_adv', cell_state_c, step_dict)
                print("cell_state_c_adv:", cell_state_c)
    elif "gate" in guided_coverage:
        if label == "nan":
            save_coverage('input_gate_c_adv', [np.nan, np.nan, np.nan, np.nan, np.nan], step_dict)
            save_coverage('new_input_c_adv', [np.nan, np.nan, np.nan, np.nan, np.nan], step_dict)
            save_coverage('forget_gate_c_adv', [np.nan, np.nan, np.nan, np.nan, np.nan], step_dict)
            save_coverage('output_gate_c_adv', [np.nan, np.nan, np.nan, np.nan, np.nan], step_dict)
        else:
            # compute the gate coverage
            layers, fw_bw, batch, steps, rnn_size = np.array(h_states_all_value).shape

            initial_states = np.zeros([layers, fw_bw, batch, 1, rnn_size])
            gates_all = np.append(initial_states, h_states_all_value, axis=3)  # [layer, fw_bw, batch, step+1, rnn_size]

            input_gate_c, new_input_c, forget_gate_c, output_gate_c = gate_coverage(gates_all, inputs, max_sen_length)
            print("input_gate_c:", input_gate_c)
            print("new_input_c:", new_input_c)
            print("forget_gate_c:", forget_gate_c)
            print("output_gate_c:", output_gate_c)

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
        if label == "nan":
            step_dict['embedding_coverage_adv'] = np.nan
        else:
            # compute the coverage like DeepXplore
            h_array = h_states_all_value.swapaxes(0, 2)  # initial: layers, fw_bw, batch, steps, rnn_size
            h_array = h_array.swapaxes(1, 2)
            h_array = h_array.swapaxes(2, 3)
            unit_coverage = unit_neuron_coverage(h_array)

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


def save_coverage(name, coverages, step_dict):
    i = 0
    for item in coverages:
        step_dict[name+'['+str(i)+']'] = item
        i += 1


def extract_c_h_data(outputs_value):
    outputs_value = np.array(outputs_value)
    last_axis = len(outputs_value.shape) - 1
    c_states_all_value, h_states_all_value = np.split(outputs_value, 2, axis=last_axis)
    return c_states_all_value, h_states_all_value


def extract_c_h_tensors(outputs_tensor):
    h_list = []
    c_list = []
    for l, layer_tuple in enumerate(outputs_tensor):
        h_list.append([])
        c_list.append([])
        for cell in layer_tuple:  # fw, bw
            c, h = tf.split(cell, num_or_size_splits=2, axis=2)
            c_list[l].append(c)
            h_list[l].append(h)
    h_states_all = tf.stack(h_list)  # layers, fw_bw, batch, steps, rnn_size
    c_states_all = tf.stack(c_list)

    return c_states_all, h_states_all


def output_seq(prefix, text):
    seq = text[0]
    pad = vocab_to_int["<PAD>"]
    eos = vocab_to_int["<EOS>"]
    out_seq = "".join([int_to_vocab[i] for i in seq if i != pad and i != eos])
    print(prefix + '{}'.format(out_seq))
    return out_seq


def predict(model, sess, adv_sentence, target, step_index, pairs, h_states_all, c_states_all):
    target_str = "".join([int_to_vocab[i] for i in target])
    target_chars = characters(target_str)

    fetches = [model.cost, model.predictions, model.rnn_outputs, model.input_embedding, model.input_reverse,
               h_states_all, c_states_all]
    cost_adv, answer_logits, rnn_outputs, input_embedding, input_reverse, h_states_all_value, c_states_all_value = \
        sess.run(fetches, {model.inputs: [adv_sentence] * batch_size,
                           model.inputs_length: [len(adv_sentence)] * batch_size,
                           model.targets: [target] * batch_size,
                           model.targets_length: [len(target)] * batch_size,
                           model.keep_prob: [1.0]})
    answer_logits = answer_logits[0]
    if step_index >= np.array(rnn_outputs).shape[1]:
        probabilities = []
    else:
        rnn_outputs = rnn_outputs[0][step_index]
        probabilities = np.exp(rnn_outputs) / sum(np.exp(rnn_outputs))

    if probabilities != []:
        # plot the changes of probabilities of original text and adversarial text
        # vis_texts(probabilities, origin_text, char_id)
        pass
    inputs = np.append(np.expand_dims(input_embedding, axis=0), np.expand_dims(input_reverse, axis=0), axis=0)

    # Remove the padding from the generated sentence
    pad = vocab_to_int["<PAD>"]
    eos = vocab_to_int["<EOS>"]
    print('  Adversarial Input Words: {}'.format("".join([int_to_vocab[i] for i in adv_sentence])))
    adv_response = "".join([int_to_vocab[i] for i in answer_logits if i != pad and i != eos])
    print('  Response Words: {}'.format(adv_response))
    print('')

    suc = is_succeeded(pairs, adv_response)

    # compute the evaluation metrics
    adv_response_chars = [characters(adv_response)]
    bleu_each = sentence_bleu(adv_response_chars, target_chars)
    WER_each = Levenshtein.distance(adv_response, target_str)

    return cost_adv, bleu_each, WER_each, h_states_all_value, c_states_all_value, inputs, suc


def predict_each(sess, model, text, target, char_id, h_states_all, c_states_all, guided_coverage, step_dict, label):
    fetches = [model.cost, model.predictions, model.rnn_outputs, model.input_embedding, model.input_reverse,
               h_states_all, c_states_all]
    cost_adv, answer_logits, rnn_outputs, input_embedding, input_reverse, h_states_all_value, c_states_all_value = \
        sess.run(fetches,{model.inputs: [text] * batch_size,
                          model.inputs_length: [len(text)] * batch_size,
                          model.targets: [target] * batch_size,
                          model.targets_length: [len(target)] * batch_size,
                          model.keep_prob: [1.0]})
    answer_logits = answer_logits[0]
    if char_id >= np.array(rnn_outputs).shape[1]:
        probabilities = []
    else:
        rnn_outputs = rnn_outputs[0][char_id]
        probabilities = np.exp(rnn_outputs) / sum(np.exp(rnn_outputs))
    inputs = np.append(np.expand_dims(input_embedding, axis=0), np.expand_dims(input_reverse, axis=0), axis=0)
    update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage, max_length, step_dict, label)
    return cost_adv, answer_logits, probabilities


def vis_texts(probabilities, text, char_id):
    idx_sort = np.argsort(probabilities)[::-1]
    word_sort = [int_to_vocab[int(x1)] for x1 in idx_sort if x1 != 78]
    prob_sort = [probabilities[x1] for x1 in idx_sort if x1 != 78]
    plot_bar(word_sort[:10], prob_sort[:10], str(int_to_vocab[text[char_id]]))


def plot_bar(word_list, prob_list, title):
    word_list = ['"' + x1 + '"' for x1 in word_list]
    rects = plt.bar(range(len(prob_list)), prob_list, color='b')
    index = range(len(word_list))
    index = [float(c) for c in index]
    plt.ylim(ymax=max(prob_list) + 0.03, ymin=0)
    plt.xticks(index, word_list)
    plt.ylabel("probability")
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 3)), ha='center', va='bottom')
    # plt.show()
    plt.title("Top 10 prediction probability for [" + title + ']')
    plt.savefig(title + '.png')
    plt.close()


def get_test_data(test_file):
    test_file = open(test_file, 'r')
    test_sequences = test_file.readlines()
    inputs = []
    targets = []
    for sequence in test_sequences:
        sequence = sequence[:-1]
        input, target = sequence.split("=>")
        inputs.append(input)
        targets.append(target)
    return inputs, targets


def characters(text):
    return [t for t in text]


def tokenize(text):
    text = re.sub(r'[{}@_*>()\\#%+=.\[\]]', '', text)
    text_tokens = text.split(" ")
    return text_tokens


def correct_pairs(text, target, ori_response):
    pairs = []
    input_tokens = tokenize(text)
    target_tokens = tokenize(target)
    ori_response_tokens = tokenize(ori_response)
    for i in input_tokens:
        if i not in target_tokens and i not in ori_response_tokens and len(i) > 1:  # wrong words but maybe corrected
            distances = []
            for r in ori_response_tokens:
                distances.append(Levenshtein.distance(i, r))

            # min_dis_index = np.argmin(distances)
            # near_word = ori_response_tokens[min_dis_index]
            distances_sorted = np.sort(distances)
            indexes = np.argsort(distances)
            min_dis = distances_sorted[0]
            min_dis_indexes = []
            for dis_i, dis in enumerate(distances_sorted):
                if dis > min_dis:
                    break
                else:
                    min_dis_indexes.append(indexes[dis_i])

            bleu_list = []
            for index in min_dis_indexes:
                near_word = ori_response_tokens[index]
                if near_word in target_tokens:  # likely to be corrected
                    bleu_list.append(sentence_bleu(characters(i), characters(near_word)))
            if len(bleu_list) > 0:
                word_index = np.argmax(bleu_list)
                pairs.append([i, ori_response_tokens[min_dis_indexes[int(word_index)]]])
    #
    # print("input_tokens")
    # print(input_tokens)
    # print("target_tokens")
    # print(target_tokens)
    # print("ori_response_tokens")
    # print(ori_response_tokens)
    print("pairs: ", pairs)
    return pairs  


def is_succeeded(pairs, adv_response):
    suc = False
    adv_response_tokens = tokenize(adv_response)
    for input_token, target_token in pairs:
        if target_token not in adv_response_tokens:
            suc = True
    return suc


def test():
    checkpoint = "./kp=0.85,nl=2,th=0.95,11.ckpt"
    keep_probability = 0.85
    num_layers = 2

    # threshold = 0.95
    model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size,
                        direction)

    if FLAGS.exp == "baseline":
        csv_filename = str(FLAGS.guided_coverage) + "_baseline_" + str(int(time.time())) + ".csv"
    elif FLAGS.objective == 1:
        csv_filename = str(FLAGS.objective) + "_" + str(FLAGS.guided_coverage) + "_adv_" + str(int(time.time())) + ".csv"
    else:
        csv_filename = str(FLAGS.objective) + "_" + str(FLAGS.objective1) + "_" + str(
            FLAGS.guided_coverage) + "_adv_" + str(int(time.time())) + ".csv"

    print("csv_filename=", csv_filename)

    with tf.Session() as sess:
        # Load saved model
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        texts, targets = get_test_data(test_file="test_seq_short")  # "test_seq"

        # csv_filename_ori = str(FLAGS.guided_coverage) + "_ori_" + ".csv"
        # if csv_filename_ori not in os.listdir("./experiment_result_boost/"):
        #     run_ori(sess, model, texts, targets, csv_filename=csv_filename_ori)
        #     # with open('./experiment_result/' + csv_filename_ori, 'a+') as f:
        #     #     writer = csv.writer(f)
        #     #     writer.writerow(['Original Test Perplexity', test_perplexity_ori])
        #     exit(0)

        start = time.clock()
        success_rate, generate_rate, generation_guided_rate = run_exp(sess, model, texts, targets, csv_filename=csv_filename)
        end = time.clock()
        duration = end - start
        print("Run time: ", duration)

        with open('./experiment_result_boost/'+csv_filename, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(['Total Run Time', duration])
            writer.writerow(['Success Rate', success_rate])
            writer.writerow(['Generation Rate', generate_rate])
            writer.writerow(['Generation Guided Rate', generation_guided_rate])

        print 'save ' + csv_filename + 'successfully'


if __name__ == "__main__":
    test()


