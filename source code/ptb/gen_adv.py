from ptb_word_lm_wrapped import *
from coverage import *  # import the spellchecker's
import myreader as reader
import random
import matplotlib.pyplot as plt
from scipy.special import expit


flags = tf.flags

flags.DEFINE_integer('topk', 3, 'topk')
flags.DEFINE_string('guided_coverage',
                    "hidden_state", 'guided_coverage,could be "hidden_state","cell_state","DX",'
                                    '"input_gate","new_input_gate","forget_gate","output_gate"')
flags.DEFINE_integer('times_nn', 3, 'num_neurons = model.num_steps * times_of_ns')
flags.DEFINE_integer('objective', 1, '0 for objective1, 1 for objective2, 2 for objective1 + objective2')
flags.DEFINE_string('objective1', "diff", 'objective1 could be "diff", "cost", "state diff"')
flags.DEFINE_integer('times_ns', 10, 'num_samples = model.batch_size * model.num_steps * 3')
flags.DEFINE_integer('lr_l', 20, "the lower bound of the learning rate")
flags.DEFINE_integer('lr_u', 100, 'the upper bound of the learning rate')
flags.DEFINE_string('exp', "baseline", 'exp could be "baseline" and "adv"')

FLAGS = flags.FLAGS


def run_adv(session, model, inverseDictionary, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    costs_adv = 0.0  # the cost for adv_sequences
    iters = 0
    state = session.run(model.initial_state)

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):

        # which time step of the input to be modified
        step_index = random.randint(1, model.num_steps - 1)
        # step_index = model.num_steps - 2
        print("------Step ", step, "-------")
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

        # for diff
        objective1 = get_objective1(model, step_index, c_states_all, h_states_all)

        fetches = [model.cost, model.initial_state, model.logits, model.inputs, model.embedding, h_states_all,
                   c_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        cost, initial_state, logits_value, inputs, embedding, h_states_all_value, c_states_all_value \
            = session.run(fetches, feed_dict)

        guided_coverage = "new_input_gate"  # could be configured
        num_neurons = model.batch_size * model.num_steps * 10
        objective2, neurons_list = neuron_selection(num_neurons, guided_coverage, h_states_all, c_states_all,
                                                    h_states_all_value, c_states_all_value)

        # objective = objective1 + objective2
        objective = objective2
        grads = tf.gradients(objective, model.inputs)

        # originally predict
        fetches = [grads, neurons_list]  #model.cost, model.initial_state, model.logits, model.inputs, model.embedding, h_states_all, c_states_all
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        grads_value, value_list = session.run(fetches, feed_dict)
        # update the coverage information after each prediction
        update_coverage(h_states_all_value, c_states_all_value, inputs, initial_state, guided_coverage)

        ori_sequences = id_to_words(x, inverseDictionary)
        print("ori_sequences")
        print(ori_sequences)

        next_batch_sequences = predict_next_batch(model, x, session, inverseDictionary)
        print('next_batch_sequences = ', next_batch_sequences)

        num_samples = model.batch_size * model.num_steps * 3

        ori_samples = id_to_words(do_sample(session, model, x, num_samples), inverseDictionary)
        print("ori_samples")
        print(ori_samples)

        # ori_res_sequences = id_to_words(predict(logits_value, x, step_index, inverseDictionary), inverseDictionary)
        # print("ori_res_sequences")
        # print(ori_res_sequences)

        x_prime = generate_adv_sequences(x, inputs, grads_value, embedding, step_index)
        adv_sequences = id_to_words(x_prime, inverseDictionary)
        print("adv_sequences")
        print(adv_sequences)

        next_adv_batch_sequences = predict_next_batch(model, x_prime, session, inverseDictionary)
        print('next_adv_batch_sequences = ', next_adv_batch_sequences)

        adv_samples = id_to_words(do_sample(session, model, x_prime, num_samples), inverseDictionary)
        print("samples after adversarial attacks")
        print(adv_samples)

        # predict after modification
        fetches = [model.cost, model.initial_state, model.logits, model.inputs, h_states_all, c_states_all]
        feed_dict = {}
        feed_dict[model.input_data] = x_prime
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        cost_adv, new_initial_state, logits_value_now, new_inputs, h_states_new_value, c_states_new_value = \
            session.run(fetches, feed_dict)
        update_coverage(h_states_new_value, c_states_new_value, new_inputs, new_initial_state, guided_coverage)
        # adv_res_sequences = id_to_words(predict(logits_value_now, x_prime, step_index, inverseDictionary),
        #                                 inverseDictionary)
        # print("adv_res_sequences")
        # print(adv_res_sequences)

        costs += cost
        print("cost:", cost)
        print("cost_adv:", cost_adv)
        costs_adv += cost_adv
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("Perplexity of the original input sequences:")
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
            print("Perplexity of the adversarial input sequences:")
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs_adv / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters), np.exp(costs_adv / iters)


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
    elif FLAGS.objective1 == "state diff":
        objective1 = h_states_all[step - 1] + c_states_all[step] - h_states_all[step]  # state diff
    else:
        print("Objective1 not configured.")
    return objective1


def update_coverage(h_states_all_value, c_states_all_value, inputs, initial_state, guided_coverage):
    # compute the state coverage
    if guided_coverage == "hidden_state":
        hidden_state_c = hidden_state_coverage(h_states_all_value)
        print("hidden_state_c:", hidden_state_c)
    elif guided_coverage == "cell_state":
        cell_state_c = cell_state_coverage(c_states_all_value)
        print("cell_state_c:", cell_state_c)
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
        print("input_gate_c:", input_gate_c)
        print("new_input_c:", new_input_c)
        print("forget_gate_c:", forget_gate_c)
        print("output_gate_c:", output_gate_c)
    elif guided_coverage == "DX":
        # compute the coverage like DeepXplore
        h_array = h_states_all_value.swapaxes(0, 2)  # batch, layer, step, embedding
        # unit_coverage = unit_neuron_coverage(h_array)
        embedding_coverage = embedding_neuron_coverage(h_array)
        # print("unit_coverage:", unit_coverage)
        print("embedding_coverage:", embedding_coverage)
    else:
        print("None coverage criteria named: ", guided_coverage)


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


def generate_adv_sequences(x, inputs, grads_value, embedding, step_index):
    batch_size, num_steps, _ = np.array(inputs).shape
    adv_sequences = np.copy(x)

    pert = grads_value[0]
    for input_index in range(batch_size):
        # just apply the perturb to the word of the step_index
        step_pert = np.array(pert[input_index][step_index]).flatten()
        step_input = np.array(inputs[input_index][step_index]).flatten()

        last_id = -1
        min_id = -1
        lr_value = 0
        for learning_rate in range(20, 70):  # (1, 50)
            # print("learning_rate:", learning_rate)
            pert_val = step_pert * learning_rate
            new_input_val = step_input + pert_val
            embedding_val = np.array(embedding)

            lr_value = learning_rate
            dist = []
            for emb in embedding_val:
                dist.append(np.linalg.norm(emb - new_input_val))
            min_id = int(np.argmin(dist))  # the word that is nearest to the new input embedding
            # print("learning rate: ", learning_rate)
            # print("min id: ", min_id)
            if last_id == -1:
                last_id = min_id  # first still be the original word
            elif last_id != min_id:
                break
        print("learning_rate: ", lr_value)
        adv_sequences[input_index][step_index] = min_id
    return adv_sequences


def test():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _, word_to_id = raw_data
    # Rani: added inverseDictionary
    inverseDictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    config = get_config()
    eval_config = get_config()
    # print(eval_config.batch_size, eval_config.num_steps)
    eval_config.batch_size = 1
    eval_config.num_steps = 10

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

        start = time.clock()

        test_perplexity, test_perplexity_adv = run_adv(session, mtest, inverseDictionary, test_data, tf.no_op())

        end = time.clock()
        duration = end-start
        print("Run time: ", duration)
        print("Original Test Perplexity: %.3f" % test_perplexity)
        print("Test Perplexity after adversarial attacks: %.3f" % test_perplexity_adv)


if __name__ == "__main__":
    test()
