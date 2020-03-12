from coverage import *
import random
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from scipy.special import expit
import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile

import scipy.io.wavfile as wav
import struct
import time
import os
import sys
import pandas as pd
import Levenshtein
import collections
import csv
import re

sys.path.append("DeepSpeech")

# necessary code for deepspeech
tf.load_op_library = lambda x: x
# print(tf.load_op_library)
tmp = os.path.exists
os.path.exists = lambda x: True
class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]
class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v
tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"


_FLOATX = 'float32'
flags = tf.flags

flags.DEFINE_string("data_path", "./testdata/", "data_path")
flags.DEFINE_integer('topk', 3, 'topk')
flags.DEFINE_string('guided_coverage', "hidden_state",
                    'guided_coverage, could be "hidden_state","cell_state","DX","input_gate","new_input_gate",'
                    '"forget_gate","output_gate"')
flags.DEFINE_integer('objective', 2, '0 for objective1, 1 for objective2, 2 for objective1 + objective2')
flags.DEFINE_string('objective1', "state_diff", 'objective1 could be "diff", "cost", "state diff"')
flags.DEFINE_integer('iters', 100, 'maximum iteration times for generating adversarial examples')
flags.DEFINE_integer('step_num', 3, 'num of steps to select for objective1')
flags.DEFINE_string('exp', "adv", 'baseline or adv')

FLAGS = flags.FLAGS


def run_adv(csv_filename):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the test dataset")

    success_num_all = 0
    tested_texts_num = 0
    with tf.Session() as sess:
        is_model_restored = False

        label_file_path = "./commonvoice/labels.csv"
        audio_label_dict, max_audio_length = load_metadata(label_file_path)

        csv_dict = []  # save the value that need to be saved to a csv file

        start = time.time()
        for audio_file in os.listdir(FLAGS.data_path):
            step_dict = collections.OrderedDict()
            step_dict['step'] = audio_file
            print("audio_file: ", audio_file)
            if audio_file.split(".")[-1] == 'wav':
                _, audio = wav.read(FLAGS.data_path + audio_file)
            else:
                raise Exception("Unknown file format")

            tested_texts_num += 1
            N = len(audio)
            length = (len(audio) - 1) // 320

            new_input = tf.placeholder(tf.float32, [1, N])
            lengths = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):  # set the graph
                logits, states_all, RNN_inputs = get_logits(new_input, lengths)

            if not is_model_restored:
                saver = tf.train.Saver()
                saver.restore(sess, "models/session_dump")
                is_model_restored = True

            c_states_all = []
            h_states_all = []
            for output in states_all:
                output_c, output_h = tf.split(output, num_or_size_splits=2, axis=2)
                c_states_all.append(output_c)
                h_states_all.append(output_h)

            # originally predict
            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

            fetches = [logits, decoded, c_states_all, h_states_all, RNN_inputs]
            logits_value, r, c_states_all_value, h_states_all_value, inputs = sess.run(fetches,
                                                                         {new_input: [audio], lengths: [length]})

            original_res = "".join([toks[x] for x in r[0].values])
            label = audio_label_dict[audio_file]
            target_chars = characters(label)

            # model performance metrics
            ori_response_chars = [characters(original_res)]
            ori_bleu = sentence_bleu(ori_response_chars, target_chars)
            ori_WER = Levenshtein.distance(original_res, label)
            print("original_res: ", original_res)
            print("ori_WER: ", ori_WER)
            print("ori_bleu: ", ori_bleu)
            # WER_ori_list.append(ori_WER)
            # bleu_ori_list.append(ori_bleu)
            step_dict["WER_ori"] = ori_WER
            step_dict["bleu_ori"] = ori_bleu

            # update coverage
            guided_coverage = FLAGS.guided_coverage
            # update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage, max_audio_length,
            #                 step_dict,  "")

            # step_index = random.randint(1, length - 1)  # it's a problem
            # step_index = length // 2

            label_phrase = tf.Variable(np.zeros((1, len(label)), dtype=np.int32))
            label_phrase_length = tf.Variable(np.zeros(1), dtype=np.int32)
            label_tensor = ctc_label_dense_to_sparse(label_phrase, label_phrase_length, 1)  # should be tensor
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(label_tensor, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            if FLAGS.objective == 0:
                step_num = FLAGS.step_num
                step_indexes = get_impact_step_index(label, original_res, length, step_num)
                objective1 = get_objective1(step_indexes, h_states_all, c_states_all, logits, logits_value, ctcloss)
                objective = objective1
            elif FLAGS.objective == 1:
                num_neurons = length // 3
                objective2, neurons_list = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all,
                                                            h_states_all_value, c_states_all_value)
                objective = objective2
            elif FLAGS.objective == 2:
                step_num = FLAGS.step_num
                step_indexes = get_impact_step_index(label, original_res, length, step_num)
                objective1 = get_objective1(step_indexes, h_states_all, c_states_all, logits, logits_value, ctcloss)
                num_neurons = length // 3
                objective2, neurons_list = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all,
                                                            h_states_all_value, c_states_all_value)
                objective = objective1 + objective2
            else:
                objective = 0
                print('Invalid input.  0 for objective1, 1 for objective2, 2 for objective1 + objective2')

            # objective1 = get_objective1(step_indexes, h_states_all, c_states_all, logits, logits_value, ctcloss)
            #
            # num_neurons = length // 3
            # objective2, neurons_list = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all,
            #                                             h_states_all_value, c_states_all_value)
            # # neurons_list = []
            # objective = objective2
            grads = tf.gradients(objective, new_input)

            label_numerical = [[toks.index(x) for x in label]]
            sess.run(label_phrase.assign(label_numerical))
            sess.run(label_phrase_length.assign(np.array([len(x) for x in label_numerical])))

            # generate adv sequences
            tmp_audio = audio
            suc = False
            for i in range(FLAGS.iters):
                grads_value, ctcloss_value, r = sess.run(
                    [grads, ctcloss, decoded],
                    {new_input: [tmp_audio], lengths: [length]})
                predicted_res = "".join([toks[x] for x in r[0].values])

                ori_norm = np.linalg.norm(audio)

                # grads_value = grads_value[0][0]
                scale = 10
                # if FLAGS.objective == 2 and FLAGS.objective1 == "state_diff" and \
                #                 FLAGS.guided_coverage in ["hidden_state", "cell_state"]:
                #     scale = 10
                # else:
                #     scale = 100
                grads_value = grads_value[0][0] * scale

                # print("grads value ori")
                # print(np.linalg.norm(grads_value)/ori_norm)

                pert_norm = np.linalg.norm(grads_value)
                distance = pert_norm / ori_norm

                print("distance")
                print(distance)

                print("ctcloss_value: ", ctcloss_value)
                WER_adv = Levenshtein.distance(predicted_res, label)
                adv_response_chars = [characters(predicted_res)]
                bleu_adv = sentence_bleu(adv_response_chars, target_chars)

                if WER_adv > ori_WER or bleu_adv < ori_bleu:
                    suc = True
                    print("Succeed, ", i)
                    print("predicted_res: ", predicted_res)
                    print("bleu_adv: ", bleu_adv)
                    print("WER_adv:{} ".format(WER_adv))
                    step_dict["WER_adv"] = WER_adv
                    step_dict["bleu_adv"] = bleu_adv
                    step_dict['bleu_dec'] = ori_bleu - bleu_adv
                    step_dict['WER_inc'] = WER_adv - ori_WER

                    step_dict['perturbation'] = distance
                    # save the adv audio
                    if not os.path.exists("./advs/" + csv_filename[:-4]):
                        os.makedirs("./advs/" + csv_filename[:-4])
                    wav.write("./advs/" + csv_filename[:-4] + "/adv_" + audio_file, 16000,
                              np.array(np.clip(np.round(tmp_audio),
                                               -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    break
                else:
                    input_adv = tmp_audio + grads_value
                    tmp_audio = input_adv

            # update coverage after each prediction, ... not if too slow
            fetches_adv = [c_states_all, h_states_all, RNN_inputs]
            c_states_all_value_adv, h_states_all_value_adv, inputs_adv = sess.run(fetches_adv,
                                                                                  {new_input: [tmp_audio],
                                                                                   lengths: [length]})

            update_coverage(h_states_all_value_adv, c_states_all_value_adv, inputs_adv, guided_coverage,
                            max_audio_length, step_dict, label="adv")

            # step_dict['adv_time'] = duration
            # step_dict['step'] = audio_file
            if suc:
                success_num_all += 1
            else:
                step_dict["WER_adv"] = WER_adv
                step_dict["bleu_adv"] = bleu_adv
                step_dict['bleu_dec'] = ori_bleu - bleu_adv
                step_dict['WER_inc'] = WER_adv - ori_WER
                step_dict['perturbation'] = distance

            print("suc: ", suc)
            step_dict_sorted = sorted(step_dict.items(), key=lambda x: x[0])
            print("step dict")
            print(dict(step_dict_sorted))
            csv_dict.append(dict(step_dict_sorted))

        end = time.time()
        duration = end - start
        print("duration: ", duration)

        success_rate = float(success_num_all) / tested_texts_num
        save_csv(csv_dict, csv_filename)
        print("success_rate: ", success_rate)
        # print("distance: mean ", np.mean(distance_list), " median ", np.median(distance_list))
        # print("iter: mean ", np.mean(iter_list), " median ", np.median(iter_list))
        # print("WER ori: mean ", np.mean(WER_ori_list), " median ", np.median(WER_ori_list))
        # print("WER adv: mean ", np.mean(WER_adv_list), " median ", np.median(WER_adv_list))
        # print("bleu ori: mean ", np.mean(bleu_ori_list), " median ", np.median(bleu_ori_list))
        # print("bleu adv: mean ", np.mean(bleu_adv_list), " median ", np.median(bleu_adv_list))
        return success_rate, duration


def run_ori(csv_filename):
    # success_num_all = 0
    tested_texts_num = 0

    with tf.Session() as sess:
        is_model_restored = False
        label_file_path = "./commonvoice/labels.csv"
        audio_label_dict, max_audio_length = load_metadata(label_file_path)

        csv_dict = []  # save the value that need to be saved to a csv file

        # start = time.time()
        for audio_file in os.listdir(FLAGS.data_path):
            step_dict = collections.OrderedDict()
            step_dict['step'] = audio_file
            print("audio_file: ", audio_file)
            if audio_file.split(".")[-1] == 'wav':
                _, audio = wav.read(FLAGS.data_path + audio_file)
            else:
                raise Exception("Unknown file format")

            tested_texts_num += 1
            N = len(audio)
            length = (len(audio) - 1) // 320

            new_input = tf.placeholder(tf.float32, [1, N])
            lengths = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):  # set the graph
                logits, states_all, RNN_inputs = get_logits(new_input, lengths)

            if not is_model_restored:
                saver = tf.train.Saver()
                saver.restore(sess, "models/session_dump")
                is_model_restored = True

            c_states_all = []
            h_states_all = []
            for output in states_all:
                output_c, output_h = tf.split(output, num_or_size_splits=2, axis=2)
                c_states_all.append(output_c)
                h_states_all.append(output_h)

            # originally predict
            fetches = [c_states_all, h_states_all, RNN_inputs]
            c_states_all_value, h_states_all_value, inputs = sess.run(fetches,
                                                                        {new_input: [audio], lengths: [length]})
            guided_coverage = FLAGS.guided_coverage
            update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage, max_audio_length, step_dict, label="ori")  # ...
            csv_dict.append(step_dict)

        save_csv(csv_dict, csv_filename)
        print('save success')


def run_baseline(csv_filename):  # random perturbation
    success_num_all = 0
    tested_texts_num = 0

    with tf.Session() as sess:
        is_model_restored = False
        label_file_path = "./commonvoice/labels.csv"
        audio_label_dict, max_audio_length = load_metadata(label_file_path)

        csv_dict = []  # save the value that need to be saved to a csv file

        start = time.time()
        for audio_file in os.listdir(FLAGS.data_path):
            step_dict = collections.OrderedDict()
            step_dict['step'] = audio_file
            print("audio_file: ", audio_file)
            if audio_file.split(".")[-1] == 'wav':
                _, audio = wav.read(FLAGS.data_path + audio_file)
            else:
                raise Exception("Unknown file format")

            tested_texts_num += 1
            N = len(audio)
            length = (len(audio) - 1) // 320

            new_input = tf.placeholder(tf.float32, [1, N])
            lengths = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):  # set the graph
                logits, states_all, RNN_inputs = get_logits(new_input, lengths)

            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

            if not is_model_restored:
                saver = tf.train.Saver()
                saver.restore(sess, "models/session_dump")
                is_model_restored = True

            c_states_all = []
            h_states_all = []
            for output in states_all:
                output_c, output_h = tf.split(output, num_or_size_splits=2, axis=2)
                c_states_all.append(output_c)
                h_states_all.append(output_h)

            # originally predict
            label = audio_label_dict[audio_file]
            target_chars = characters(label)

            r = sess.run(decoded, {new_input: [audio], lengths: [length]})
            original_res = "".join([toks[x] for x in r[0].values])

            ori_response_chars = [characters(original_res)]
            ori_bleu = sentence_bleu(ori_response_chars, target_chars)
            ori_WER = Levenshtein.distance(original_res, label)

            # generate baseline audio
            mu = 100
            sigma = 10
            random_pert = np.random.normal(mu, sigma, audio.shape)
            base_audio = audio + random_pert

            fetches = [c_states_all, h_states_all, RNN_inputs, decoded]
            c_states_all_value_base, h_states_all_value_base, inputs_base, r_base = sess.run(fetches,
                                                                       {new_input: [base_audio], lengths: [length]})

            res_base = "".join([toks[x] for x in r_base[0].values])
            base_chars = [characters(res_base)]
            bleu_base = sentence_bleu(base_chars, target_chars)
            WER_base = Levenshtein.distance(res_base, label)

            print("predicted_res: ", res_base)
            print("bleu_base: ", bleu_base)
            print("WER_base:{} ".format(WER_base))
            step_dict["WER_base"] = WER_base
            step_dict["bleu_base"] = bleu_base
            step_dict['bleu_dec'] = ori_bleu - bleu_base
            step_dict['WER_inc'] = WER_base - ori_WER

            pert_norm = np.linalg.norm(random_pert)
            ori_norm = np.linalg.norm(audio)
            distance = pert_norm / ori_norm
            # distance_list.append(distance)
            # iter_list.append(i)
            step_dict['perturbation'] = distance
            print("distance")
            print(distance)

            # to predict
            if WER_base > ori_WER or bleu_base < ori_bleu:
                print("Succeed")
                # save the adv audio
                if not os.path.exists("./advs/" + csv_filename[:-4] + "base"):
                    os.makedirs("./advs/" + csv_filename[:-4] + "base")
                wav.write("./advs/" + csv_filename[:-4] + "base" + "/base_adv_" + audio_file, 16000,
                          np.array(np.clip(np.round(base_audio),
                                           -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                success_num_all += 1

            guided_coverage = FLAGS.guided_coverage
            update_coverage(h_states_all_value_base, c_states_all_value_base, inputs_base, guided_coverage,
                            max_audio_length, step_dict, label="base")
            csv_dict.append(step_dict)

        end = time.time()
        duration = end - start
        success_rate = success_num_all / tested_texts_num

        save_csv(csv_dict, csv_filename)
        print('save success')
        return success_rate, duration


def get_impact_step_index(label, original_res, step_length, step_nums):
    step_indexes = []

    label_tokens = tokenize(label)
    original_res_tokens = tokenize(original_res)

    for tok_res in original_res_tokens:
        if tok_res in label_tokens:
            tok_index = original_res.find(tok_res)
            step_index = round(step_length / len(original_res) * tok_index)
            step_indexes.append(step_index)
            print("select: ", step_index)
            if len(step_indexes) >= step_nums:
                break
    if len(step_indexes) < step_nums:
        remain = step_nums - len(step_indexes)
        for i in range(remain):
            step_index = step_length // (remain + 1) * (i + 1)  # just select the middle step
            step_indexes.append(step_index)
            print("remain: ", step_index)
    print("step indexes")
    print(step_indexes)
    return step_indexes


# def get_impact_step_index(label, original_res, step_length, step_nums):
#     # print("label: ", label)
#     # print("original res: ", original_res)
#     step_indexes = []
#     tok_indexes = []
#
#     tok_index = -1
#     label_tokens = tokenize(label)
#     original_res_tokens = tokenize(original_res)
#     # print("label tokens: ", label_tokens)
#     # print("original res tokens: ", original_res_tokens)
#     for tok_res in original_res_tokens:
#         if tok_res in label_tokens:
#             tok_index = original_res.find(tok_res)
#             # print("tokens: ", tok_res)
#             break
#     if tok_index != -1:
#         step_index = round(step_length / len(original_res) * tok_index)
#     else:
#         step_index = step_length // 2  # just select the middle step
#
#     print("step_length: ", step_length)
#     # print("tok_index: ", tok_index)
#     # print("step index: ", step_index)
#     return step_indexes


def tokenize(text):
    text = re.sub(r'[{}@_*>()\\#%+=.\[\]]', '', text)
    text_tokens = text.split(" ")
    return text_tokens


def save_csv(dict, filename):
    if os.path.exists('./experiment_result'):
        pass
    else:
        os.makedirs('./experiment_result')
    filename = './experiment_result/'+filename
    with open(filename, 'a+') as f:
        w = csv.writer(f)
        fields_names = dict[0].keys()
        w.writerow(fields_names)
        for row in dict:
            w.writerow(row.values())


def save_coverage(name, coverages, step_dict):
    i = 0
    for item in coverages:
        step_dict[name+'['+str(i)+']'] = item
        i += 1


def load_metadata(label_file_path):
    with open(label_file_path) as csv_file:
        csv_data = pd.read_csv(csv_file)
        keys = np.array(csv_data.loc[:, ['audio']])
        keys = keys.reshape([1, keys.shape[0]])[0]  # convert the frame to a list
        values = np.array(csv_data.loc[:, ['label']])
        values = values.reshape([1, values.shape[0]])[0]
        audio_label_dict = dict(zip(keys, values))

        nums = np.array(csv_data.loc[:, ['num']])
        nums = nums.reshape([1, nums.shape[0]])[0]
        max_N = (max(nums) - 44) // 2  # the correspondence of num and len(audio)
        max_audio_length = (max_N - 1) // 320
    return audio_label_dict, max_audio_length


def get_objective1(steps, h_states_all, c_states_all, logits, logits_value, ctcloss):
    objective1 = 0
    if FLAGS.objective1 == "diff":
        for step in steps:
            outputs = logits_value[step][0]  # 0: batch=1, audio input one by one
            top_idx = np.argsort(outputs)[::-1]
            for i in range(1, FLAGS.topk):
                objective1 += mean(logits[step][0][top_idx[i]])

            objective1 -= mean(logits[step][0][top_idx[0]])

    elif FLAGS.objective1 == "cost":
        objective1 = ctcloss
    elif FLAGS.objective1 == "state_diff":
        # fw_bw, steps, batch, rnn_size -> steps, fw_bw, batch, rnn_size
        h_states_all = tf.transpose(h_states_all, perm=[1, 0, 2, 3])
        c_states_all = tf.transpose(c_states_all, perm=[1, 0, 2, 3])
        for step in steps:
            objective1 = h_states_all[step - 1] + c_states_all[step] - h_states_all[step]  # state diff
    else:
        print("Objective1 not configured.")
    return objective1


# def get_objective1(step, h_states_all, c_states_all, logits, logits_value, ctcloss):
#     objective1 = 0
#     if FLAGS.objective1 == "diff":
#         outputs = logits_value[step][0]  # 0: batch=1, audio input one by one
#         top_idx = np.argsort(outputs)[::-1]
#         for i in range(1, FLAGS.topk):
#             objective1 += mean(logits[step][0][top_idx[i]])
#
#         objective1 -= mean(logits[step][0][top_idx[0]])
#
#     elif FLAGS.objective1 == "cost":
#         objective1 = ctcloss
#     elif FLAGS.objective1 == "state_diff":
#         # fw_bw, steps, batch, rnn_size -> steps, fw_bw, batch, rnn_size
#         h_states_all = tf.transpose(h_states_all, perm=[1, 0, 2, 3])
#         c_states_all = tf.transpose(c_states_all, perm=[1, 0, 2, 3])
#         objective1 = h_states_all[step - 1] + c_states_all[step] - h_states_all[step]  # state diff
#     else:
#         print("Objective1 not configured.")
#     return objective1


def mean(x, axis=None, keepdims=False):
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    return tf.reduce_mean(x, axis, keepdims)


def update_coverage(h_states_all_value, c_states_all_value, inputs, guided_coverage, max_sen_length, step_dict, label):
    # compute the state coverage
    if guided_coverage == "hidden_state":
        hidden_state_c = hidden_state_coverage(h_states_all_value, max_sen_length)
        if label == "ori":
            step_dict['hidden_state_c_ori'] = str(hidden_state_c)
        elif label == "base":
            step_dict['hidden_state_c_base'] = str(hidden_state_c)
        elif label == "adv":
            step_dict['hidden_state_c_adv'] = str(hidden_state_c)
        print("hidden_state_c:", hidden_state_c)
    elif guided_coverage == "cell_state":
        cell_state_c = cell_state_coverage(c_states_all_value, max_sen_length)
        if label == "ori":
            print("cell_state_c_ori:", cell_state_c)
            save_coverage('cell_state_c_ori', cell_state_c, step_dict)
        elif label == "base":
            save_coverage('cell_state_c_base', cell_state_c, step_dict)
        elif label == "adv":
            save_coverage('cell_state_c_adv', cell_state_c, step_dict)
            print("cell_state_c_adv:", cell_state_c)
        print("cell_state_c:", cell_state_c)
    elif "gate" in guided_coverage:
        # compute the gate coverage
        fw_bw, steps, batch, rnn_size = np.array(h_states_all_value).shape

        initial_states = np.zeros([fw_bw, 1, batch, rnn_size])
        gates_all = np.append(initial_states, h_states_all_value, axis=1)  # [fw_bw, steps+1, batch, rnn_size]

        input_gate_c, new_input_c, forget_gate_c, output_gate_c = gate_coverage(gates_all, inputs, max_sen_length)
        if label == "ori":
            print("input_gate_c_ori:", input_gate_c)
            print("new_input_c_ori:", new_input_c)
            print("forget_gate_c_ori:", forget_gate_c)
            print("output_gate_c_ori:", output_gate_c)
            save_coverage('input_gate_c_ori', input_gate_c, step_dict)
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
        h_array = np.array(h_states_all_value).swapaxes(0, 2)  # batch, steps, fw_bw, embedding
        unit_coverage = unit_neuron_coverage(h_array, max_sen_length)

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


def characters(text):
    return [t for t in text]


if __name__ == "__main__":
    csv_filename_ori = str(FLAGS.guided_coverage) + "_ori_" + ".csv"
    # if csv_filename_ori not in os.listdir("./experiment_result/"):
    #     run_ori(csv_filename_ori)

    if FLAGS.exp == "baseline":
        csv_filename = str(FLAGS.guided_coverage) + "_baseline_" + str(int(time.time())) + ".csv"
        success_rate, duration = run_baseline(csv_filename)
    elif FLAGS.objective == 1:
        csv_filename = str(FLAGS.objective) + "_" + str(FLAGS.guided_coverage) + "_adv_" + str(int(time.time())) + ".csv"
        success_rate, duration = run_adv(csv_filename)
    else:
        csv_filename = str(FLAGS.objective) + "_" + str(FLAGS.objective1) + "_" + str(
            FLAGS.guided_coverage) + "_adv_" + str(int(time.time())) + ".csv"
        success_rate, duration = run_adv(csv_filename)

    with open('./experiment_result/' + csv_filename, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['Total Run Time', duration])
        writer.writerow(['Success Rate', success_rate])
    print('save ' + csv_filename + 'successfully')
    #
    # csv_filename_ori = str(FLAGS.guided_coverage) + "_ori_" + ".csv"
    # if csv_filename_ori not in os.listdir("./experiment_result/"):
    #     run_ori(csv_filename_ori)
