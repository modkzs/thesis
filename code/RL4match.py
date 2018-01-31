# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time


class MatchModel(object):
    # TODO: change seed to current time when finish
    def __init__(self, word_vec_len, max_sen_len, seed=0):
        # all model parameter
        self.max_sen_len = max_sen_len
        self.word_evc_len = word_vec_len
        self.actions = [[0, 1], [1, 0], [1, 1]]
        self.seed = seed

        # deep match parameter
        self.deep_hidden_num = 128
        self.deep_lr = 0.001

        # reward parameter
        self.reward_hidden_num = 128
        self.reward_lr = 0.001

        # policy parameter
        self.policy_lr = 0.001

        self.deep_model = self.deep_match_model(self.deep_lr)
        self.policy = self.policy_model(self.policy_lr)
        self.reward = self.reward_model(self.reward_lr)

        self.deep_model_trained = False

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        random.seed(seed)

    def deep_match_model(self, lr):
        with tf.variable_scope('deep_match_model'):
            input_1 = tf.placeholder(shape=[None, None, self.word_evc_len], dtype=tf.float32, name="q1")
            input_2 = tf.placeholder(shape=[None, None, self.word_evc_len], dtype=tf.float32, name="q2")
            label = tf.placeholder(shape=[None, 1], dtype=np.float32, name="label")

            outputs_1, states_1 = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.deep_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE,
                                        initializer=tf.orthogonal_initializer(dtype=tf.float32, seed=self.seed)),
                input_1, dtype=tf.float32, time_major=True)
            outputs_2, states_2 = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.deep_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE,
                                        initializer=tf.orthogonal_initializer(dtype=tf.float32, seed=self.seed)),
                input_2, dtype=tf.float32, time_major=True)
            final_output = tf.concat([outputs_1[-1], outputs_2[-1]], axis=1)

            weight = tf.get_variable('weight', shape=[self.deep_hidden_num * 2, 1],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            bias = tf.get_variable('bias', shape=[1, 1],
                                   initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            logits = tf.matmul(final_output, weight) + bias
            prediction = tf.nn.sigmoid(logits)

            loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss_op)

            correct_pred = tf.equal(tf.cast(tf.greater(prediction, 0.5), tf.float32), label)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            return {"outputs_1": outputs_1, "outputs_2": outputs_2, "train_op": train_op,
                    "accuracy": accuracy, "input_1": input_1, "input_2": input_2, "label": label}

    @staticmethod
    def shuffle_data(dataset_x, dataset_y, label, seq=None):
        """
        shuffle two sentence set
        input data is already batched, which is list of [batch, ?, word_vec]
        """
        assert len(dataset_x) == len(dataset_y)
        batch_size = dataset_x[0].shape[0]

        if seq is None:
            seq = []

            last_data = None
            for d in dataset_x:
                if last_data is None or last_data.shape != d.shape:
                    last_data = d
                    seq.append(list(range(d.shape[0])))
                    continue

                seq[-1].extend(list(range(seq[-1][-1] + 1, seq[-1][-1] + 1 + d.shape[0])))

        new_data_x = []
        new_data_y = []
        labels = []

        for s in seq:
            np.random.shuffle(s)

            data_x = [dataset_x[i / batch_size][i % batch_size] for i in s]
            new_data_x.append(data_x[i:i + batch_size] for i in range(0, len(data_x), batch_size))

            data_y = [dataset_y[i / batch_size][i % batch_size] for i in s]
            new_data_y.append(data_y[i:i + batch_size] for i in range(0, len(data_y), batch_size))

            cur_label = [label[i / batch_size][i % batch_size] for i in s]
            labels.append(cur_label[i:i + batch_size] for i in range(0, len(data_y), batch_size))

        return new_data_x, new_data_y, labels

    # TODO: add data shuffle each epoch
    def train_deep_model(self, train_sen_1, train_sen_2, train_y, test_sen_1, test_sen_2, test_y, epochs=20):
        """
        train a deep match model, sentence input must be [batch, word_len, word_vec]
        word_len don't need to be same each batch
        """
        train_sen_1 = [np.transpose(x, (1, 0, 2)) for x in train_sen_1]
        train_sen_2 = [np.transpose(x, (1, 0, 2)) for x in train_sen_2]

        test_sen_1 = np.transpose(test_sen_1, (1, 0, 2))
        test_sen_2 = np.transpose(test_sen_2, (1, 0, 2))

        for k in range(epochs):
            acc = 0
            for i in range(len(train_sen_1)):
                tmp_x_1 = train_sen_1[i]
                tmp_x_2 = train_sen_2[i]
                tmp_y = train_y[i].reshape([-1, 1])
                acc += self.session.run([self.deep_model['train_op'], self.deep_model['accuracy']],
                                        feed_dict={self.deep_model['input_1']: tmp_x_1,
                                                   self.deep_model['input_2']: tmp_x_2,
                                                   self.deep_model['label']: tmp_y})[-1]

            print('train acc: %.6f' % (acc / len(train_sen_1)),
                  'test  acc: %.6f' % self.session.run(self.deep_model['accuracy'],
                                                       feed_dict={self.deep_model['input_1']: test_sen_1,
                                                                  self.deep_model['input_2']: test_sen_2,
                                                                  self.deep_model['label']: test_y}))

        self.deep_model_trained = True

    def gen_seq(self, sen_vec):
        """
        get the embedding vector of a sentence, input must be [batch, sen_len, word_vec] or [sen_len, word_vec]
        """
        if len(sen_vec.shape) < 3:
            sen_vec = sen_vec.reshape([1, sen_vec.shape[0], sen_vec.shape[1]])

        sen_vec = np.transpose(sen_vec, (1, 0, 2))
        return self.session.run(self.deep_model['outputs_1'], feed_dict={self.deep_model['input_1']: sen_vec})

    def policy_model(self, lr):
        with tf.variable_scope('policy_model'):
            sen_1 = tf.placeholder(shape=[None, self.deep_hidden_num], dtype=np.float32, name="sen1")
            sen_2 = tf.placeholder(shape=[None, self.deep_hidden_num], dtype=np.float32, name="sen2")

            vec_1 = tf.placeholder(shape=[None, self.word_evc_len], dtype=np.float32, name="vec1")
            vec_2 = tf.placeholder(shape=[None, self.word_evc_len], dtype=np.float32, name="vec2")

            # TODO: is concat too simple?
            final_vec = tf.concat([sen_1, sen_2, vec_1, vec_2], axis=1)

            weight = tf.get_variable('weight', shape=[(self.word_evc_len + self.deep_hidden_num) * 2, 3],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            bias = tf.get_variable('bias', shape=[3, ],
                                   initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            logits = tf.matmul(final_vec, weight) + bias
            prediction = tf.nn.softmax(logits)

            label = tf.placeholder(shape=[None, 3], dtype=np.float32, name='label')

            gradient = tf.placeholder(shape=[None, 1], dtype=np.float32, name='gradient')
            loss_op = tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label), 0)
            # gradient, shape is [batch_size, 1]
            loss_op = tf.matmul(loss_op, gradient)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss_op)

            return {'sen_1': sen_1, 'sen_2': sen_2, 'vec_1': vec_1, 'vec_2': vec_2, 'label': label,
                    'prediction': prediction, 'optimizer': train_op, 'reward': gradient}

    # hidden of sen_1 and sen_2 from DeepMatch model, and 2 word vector
    def get_policy(self, sen_1, sen_2, vec_1, vec_2):
        def add_dim(x): return x if len(x.shape) > 1 else x.reshape([1, -1])

        sen_1 = add_dim(sen_1)
        sen_2 = add_dim(sen_2)
        vec_1 = add_dim(vec_1)
        vec_2 = add_dim(vec_2)

        return self.session.run(self.policy['prediction'],
                                feed_dict={self.policy['sen_1']: sen_1, self.policy['sen_2']: sen_2,
                                           self.policy['vec_1']: vec_1, self.policy['vec_2']: vec_2})

    # how to update?
    def update_policy(self, reward, seq, sen_1, sen_2, vec_1, vec_2, batch_size=1024):
        # TODO: run 1 epoch or more?
        for i in range(0, len(reward), batch_size):
            self.session.run(self.policy['optimizer'], {self.policy['reward']: reward[i:i + batch_size],
                                                        self.policy['vec_1']: vec_1[i:i + batch_size],
                                                        self.policy['vec_2']: vec_2[i:i + batch_size],
                                                        self.policy['sen_1']: sen_1[i:i + batch_size],
                                                        self.policy['sen_2']: sen_2[i:i + batch_size],
                                                        self.policy['label']: seq[i:i + batch_size]})

    # TODO: This train is slow, need to rewrite it, the execute cell is 3 LSTM cell not a LSTM sentence
    def reward_model(self, lr):
        with tf.variable_scope('reward_model'):
            # sentence input
            sen = tf.placeholder(shape=[None, None, self.word_evc_len], dtype=np.float32, name="q1")
            # action seq input, each input is 3d, corresponding to 3 prob
            seqs = tf.placeholder(shape=[None, None, 3], dtype=np.float32, name="seq")
            # final label
            label = tf.placeholder(shape=[None, 1], dtype=np.float32, name="label")

            # gather index input
            map_1 = tf.placeholder(shape=[None, None, 1], dtype=np.int32, name="map_1")
            map_2 = tf.placeholder(shape=[None, None, 1], dtype=np.int32, name="map_2")

        # TODO: using this RNN is very slow. How good? may other choice(like using pre-train RNN or just word_vec)?
        with tf.variable_scope('reward_model_sen'):
            sen = tf.transpose(sen, perm=[1, 0, 2])

            # using LSTM to encode input sentence, output should be input of seq model
            outputs, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.deep_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE,
                                        initializer=tf.orthogonal_initializer(dtype=tf.float32, seed=self.seed)), sen,
                dtype=tf.float32, time_major=True)

            # mean is useless when predict, only used to compute gradient
            outputs = tf.stack(outputs)
            # first dimension is batch size
            # mean = tf.reduce_sum(outputs, [1, 2])

            gradient = tf.placeholder(dtype=np.float32)

            optimizer_sen = tf.train.RMSPropOptimizer(learning_rate=lr)
            train_op_sen = optimizer_sen.minimize(tf.reduce_mean(tf.multiply(gradient, outputs)))

        with tf.variable_scope('reward_model_seq'):
            # encode seq, the final output
            seq_sen_1 = tf.placeholder(shape=[None, None, self.reward_hidden_num], dtype=np.float32, name="q1")
            seq_sen_2 = tf.placeholder(shape=[None, None, self.reward_hidden_num], dtype=np.float32, name="q2")

            # The code below is used to extend a array [range(a)] to shape [a, b, 1](same as map_1),
            # which all [b, 1] with same a is same value. This three code act like below:
            # idx = list(range(map_1.shape[0]))
            # y = [1, map_1.shape[1]]
            # y = [[idx[i]] * y[i] for i in range(len(idx))]
            idx = tf.range(tf.gather(tf.shape(map_1), 0))
            expand_idx = tf.reshape(
                tf.stack([tf.constant([1]), tf.reshape(tf.gather(tf.shape(map_1), 1), [1])], axis=1), [-1])
            expand_idx = tf.expand_dims(tf.tile(tf.expand_dims(idx, -1), expand_idx), -1)

            seq_sen_var_1 = tf.gather_nd(seq_sen_1, tf.concat([expand_idx, map_1], 2))
            seq_sen_var_2 = tf.gather_nd(seq_sen_2, tf.concat([expand_idx, map_2], 2))

            # TODO: concat?
            seq_input = tf.concat([seq_sen_var_1, seq_sen_var_2, seqs], axis=2)
            # TODO: may find a way to avoid?
            seq_input = tf.transpose(seq_input, perm=[1, 0, 2])

            outputs_seq, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.deep_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE,
                                        initializer=tf.orthogonal_initializer(dtype=tf.float32, seed=self.seed)), seq_input,
                dtype=tf.float32, time_major=True)

            weight = tf.get_variable('weight', shape=[self.reward_hidden_num, 1],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            bias = tf.get_variable('bias', shape=[1, 1],
                                   initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))

            logits = tf.matmul(outputs_seq[-1], weight) + bias
            prediction = tf.nn.sigmoid(logits)

            loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))
            optimizer_seq = tf.train.RMSPropOptimizer(learning_rate=lr)
            train_op_seq = optimizer_seq.minimize(loss_op)

            gradient_1 = tf.reduce_mean(tf.gradients(loss_op, seq_sen_var_1), [1, 2])
            gradient_2 = tf.reduce_mean(tf.gradients(loss_op, seq_sen_var_2), [1, 2])

        return {'sen': sen, 'seqs': seqs, 'outputs': outputs, 'gradient': gradient, 'prediction': prediction,
                'train_op_seq': train_op_seq, 'train_op_sen': train_op_sen,
                'seq_sen_1': seq_sen_1, 'seq_sen_2': seq_sen_2, 'label': label,
                'sen_gradient_1': gradient_1, 'sen_gradient_2': gradient_2,
                'map_1': map_1, 'map_2': map_2}

    def get_reward(self, sen_1, sen_2, seqs, map_1, map_2):
        sen_1 = sen_1 if len(sen_1.shape) == 3 else sen_1.reshape([1, sen_1.shape[0], sen_1.shape[1]])
        sen_2 = sen_2 if len(sen_2.shape) == 3 else sen_2.reshape([1, sen_2.shape[0], sen_2.shape[1]])

        map_1 = map_1 if len(map_1.shape) == 3 else map_1.reshape([map_1.shape[0], map_1.shape[1], 1])
        map_2 = map_2 if len(map_2.shape) == 3 else map_2.reshape([map_2.shape[0], map_2.shape[1], 1])

        seqs = seqs if len(seqs.shape) == 3 else seqs.reshape([1, seqs.shape[0], seqs.shape[1]])

        output_1 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_1})
        output_2 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_2})

        return self.session.run(self.reward['prediction'],
                                feed_dict={self.reward['seq_sen_1']: output_1, self.reward['seq_sen_2']: output_2,
                                           self.reward['seqs']: seqs,
                                           self.reward['map_1']: map_1, self.reward['map_2']: map_2})

    # a really fucking complicated function.
    # Must compte reward seq model first, then get each gradient to reward sentence model, then update sentence model
    def update_reward(self, sen_1, sen_2, seqs, labels, map_1, map_2):
        sen_1 = sen_1 if len(sen_1.shape) == 3 else sen_1.reshape([1, sen_1.shape[0], sen_1.shape[1]])
        sen_2 = sen_2 if len(sen_2.shape) == 3 else sen_2.reshape([1, sen_2.shape[0], sen_2.shape[1]])
        seqs = seqs if len(seqs.shape) == 3 else seqs.reshape([1, seqs.shape[0], seqs.shape[1]])
        map_1 = map_1 if len(map_1.shape) == 3 else map_1.reshape([map_1.shape[0], map_1.shape[1], 1])
        map_2 = map_2 if len(map_2.shape) == 3 else map_2.reshape([map_2.shape[0], map_2.shape[1], 1])

        if len(labels.shape) == 1:
            labels = labels.reshape([1, 1])

        # TODO: output_1 and output_2 will go through GPU->CPU->GPU if runs on GPU. May change tensorflow to fit it?
        output_1 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_1})
        output_2 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_2})

        grad = self.session.run(
            [self.reward['train_op_seq'], self.reward['sen_gradient_1'], self.reward['sen_gradient_2']],
            feed_dict={self.reward['seq_sen_1']: output_1, self.reward['seq_sen_2']: output_2,
                       self.reward['seqs']: seqs, self.reward['label']: labels,
                       self.reward['map_1']: map_1, self.reward['map_2']: map_2}
        )[1:]

        # TODO: each cell get same weight, actually should use map to get cell use weight and using it update model?
        self.session.run(self.reward['train_op_sen'],
                         feed_dict={self.reward['sen']: sen_1, self.reward['gradient']: grad[0]})
        self.session.run(self.reward['train_op_sen'],
                         feed_dict={self.reward['sen']: sen_2, self.reward['gradient']: grad[1]})

    def fill(self, vec, length):
        return np.array(vec + [np.array([0] * self.word_evc_len)] * (length - len(vec)))

    class SimulationPath:
        """
        simulation paths with same length
        """

        def __init__(self):
            self.map_1 = []
            self.map_2 = []
            self.direction = []
            self.policy_label = []
            self.reward_label = []

            self.vec_1 = []
            self.vec_2 = []
            self.sen_1 = []
            self.sen_2 = []

            self.reward = None

        def add_path(self, add_map_1, add_map_2, add_direction, add_policy_label, add_vec_1, add_vec_2,
                     add_sen_1, add_sen_2, add_reward_label):
            self.map_1.append(add_map_1)
            self.map_2.append(add_map_2)
            self.direction.append(add_direction)
            self.policy_label.append(add_policy_label)
            self.reward_label.append(add_reward_label)

            # those np array is reference, so no memory waste
            self.vec_1.append(add_vec_1)
            self.vec_2.append(add_vec_2)
            self.sen_1.append(add_sen_1)
            self.sen_2.append(add_sen_2)

            return self

        def merge(self, sp):
            self.map_1.extend(sp.map_1)
            self.map_2.extend(sp.map_2)
            self.direction.extend(sp.direction)
            self.policy_label.extend(sp.policy_label)
            self.reward_label.extend(sp.reward_label)

            self.vec_1.extend(sp.vec_1)
            self.vec_2.extend(sp.vec_2)
            self.sen_1.extend(sp.sen_1)
            self.sen_2.extend(sp.sen_2)

            del sp

        def update_reward(self, cur_reward):
            self.reward = cur_reward

    def monte_carlo(self, vec_1, vec_2, sen_1, sen_2, gen_num, label):
        """
        monte carlo sample on a single sentence, may change to batch in future?
        :param sen_2: input sentence, [sen_len, embedding_len], current get by a deep match rnn
        :param sen_1: input sentence, [sen_len, embedding_len]
        :param label: the pair label, match or mismatch
        :param vec_1: input word vec, [sen_len, word_vec]
        :param vec_2: input word vec, [sen_len, word_vec]
        :param gen_num: generate path number
        :return:
        """
        assert vec_1.shape[0] == sen_1.shape[0]
        assert vec_2.shape[0] == sen_2.shape[0]

        len_1 = sen_1.shape[0]
        len_2 = sen_2.shape[0]

        results = {}

        # pos 2 probability map. Prob has no thing to do with past path, so can store
        # pos_p = {}

        sens_1 = []
        sens_2 = []
        vecs_1 = []
        vecs_2 = []

        for i in range(len_1):
            for j in range(len_2):
                sens_1.append(sen_1[i])
                sens_2.append(sen_2[i])
                vecs_1.append(vec_1[i])
                vecs_2.append(vec_2[i])

        probs = self.get_policy(np.array(sens_1), np.array(sens_2), np.array(vecs_1), np.array(vecs_2))

        for i in range(gen_num):
            direction = []
            map_1 = []
            map_2 = []
            labels = []

            pos = [0, 0]
            while pos[0] < len_1 and pos[1] < len_2:
                map_1.append(pos[0])
                map_2.append(pos[1])

                p = probs[pos[0] * len_2 + pos[1]]

                # if (pos[0], pos[1]) not in pos_p:
                #     p = self.get_policy(sen_1[pos[0]], sen_2[pos[1]], vec_1[pos[0]], vec_2[pos[1]])[0]
                #     pos_p[(pos[0], pos[1])] = p
                # else:
                #     p = pos_p[(pos[0], pos[1])]

                prob = random.random()
                action = int(prob > p[0]) + int(prob > p[0] + p[1])
                labels.append([int(action == 0), int(action == 1), int(action == 2)])

                pos[0] += self.actions[action][0]
                pos[1] += self.actions[action][1]

                direction.append(p)

            if pos[0] == len_1:
                direction += [np.array([1, 0, 0])] * (len_2 - pos[1])
                labels += [[1, 0, 0]] * (len_2 - pos[1])
                map_1 += [map_1[-1]] * (len_2 - pos[1])
                map_2 += [i for i in range(pos[1], len_2)]
            else:
                direction += [np.array([0, 1, 0])] * (len_1 - pos[0])
                labels += [[0, 1, 0]] * (len_1 - pos[0])
                map_1 += [i for i in range(pos[0], len_1)]
                map_2 += [map_2[-1]] * (len_1 - pos[0])

            results[len(map_1)] = results.get(len(map_1), self.SimulationPath()) \
                .add_path(map_1, map_2, direction, labels, vec_1, vec_2, sen_1, sen_2, label)
        return results

    # TODO: how to run in batch?
    def predict(self, vec_1, vec_2, sen_1, sen_2):
        len_1 = sen_1.shape[0]
        len_2 = sen_2.shape[0]

        pos = [0, 0]
        direction = []
        map_1 = []
        map_2 = []

        while pos[0] < len_1 and pos[1] < len_2:
            map_1.append(pos[0])
            map_2.append(pos[1])

            p = self.get_policy(sen_1[pos[0]], sen_2[pos[1]], vec_1[pos[0]], vec_2[pos[1]])[0]

            if p[0] > p[1] and p[0] > p[2]:
                action = 0
            elif p[1] > p[2] and p[1] > p[0]:
                action = 1
            else:
                action = 2

            pos[0] += self.actions[action][0]
            pos[1] += self.actions[action][1]

            direction.append(p)

        if pos[0] == len_1:
            direction += [np.array([1, 0, 0])] * (len_2 - pos[1])
            map_1 += [map_1[-1]] * (len_2 - pos[1])
            map_2 += [i for i in range(pos[1], len_2)]
        else:
            direction += [np.array([0, 1, 0])] * (len_1 - pos[0])
            map_1 += [i for i in range(pos[0], len_1)]
            map_2 += [map_2[-1]] * (len_1 - pos[0])

        return self.get_reward(np.array(vec_1), np.array(vec_2), np.array(direction),
                               np.array(map_1).reshape([1, -1]), np.array(map_2).reshape([1, -1]))

    # TODO: wait to write.....
    def validate(self, sen_1, sen_2, vec_1, vec_2, label, threshold=0.5, val_func=lambda t, y: (t == y).sum() / len(t)):
        preds = []
        for i in range(len(sen_1)):
            pred = int(self.predict(vec_1[i], vec_2[i], sen_1[i], sen_2[i]) > threshold)
            preds.append(pred)

        return val_func(np.array(preds).reshape(-1), label.reshape(-1))

    # TODO: all vec and sen passed in is not time major, have to transpose in tensorflow, fix it?
    def train(self, vec_1, vec_2, sen_1, sen_2, labels, simulation_num, test_vec_1, test_vec_2, test_sen_1, test_sen_2,
              test_label, epochs=100, print_epoch=1):
        """
        train RL model for match. sen must be list of [batch, ?, word_vec]
        simulation_num is generated seq number when monte carlo simulation
        """
        # iterator nums
        for fc in range(epochs):
            # TODO: current reward and policy update is running each batch, not whole epoch, may better?
            for j in range(len(vec_1)):
                batch_size = vec_1[j].shape[0]
                result = {}

                for b in range(batch_size):
                    b_result = self.monte_carlo(
                        vec_1[j][b], vec_2[j][b], sen_1[j][b], sen_2[j][b], simulation_num, labels[j][b])
                    for (k, v) in b_result.items():
                        if k in result:
                            result[k].merge(v)
                        else:
                            result[k] = v

                # TODO: currently each epoch has different data number, consider merge small length data together?
                for sp in result.values():
                    self.update_reward(np.array(sp.vec_1), np.array(sp.vec_2), np.array(sp.direction),
                                       np.array(sp.reward_label), np.array(sp.map_1), np.array(sp.map_2))

                for sp in result.values():
                    sp.update_reward(self.get_reward(np.array(sp.vec_1), np.array(sp.vec_2),
                                                     np.array(sp.direction), np.array(sp.map_1), np.array(sp.map_2)))

                # batch policy update
                policy_reward = []
                policy_sen_1 = []
                policy_sen_2 = []
                policy_vec_1 = []
                policy_vec_2 = []
                policy_label = []
                for sp in result.values():
                    for s in range(len(sp.map_2)):
                        cur_map_1 = sp.map_1[s]
                        cur_map_2 = sp.map_2[s]
                        for w in range(len(sp.map_2[s])):
                            policy_reward.append(sp.reward[s])
                            policy_vec_1.append(sp.vec_1[s][cur_map_1[w]])
                            policy_vec_2.append(sp.vec_2[s][cur_map_2[w]])
                            policy_label.append(sp.policy_label[s][w])
                            policy_sen_1.append(sp.sen_1[s][cur_map_1[w]])
                            policy_sen_2.append(sp.sen_2[s][cur_map_2[w]])

                self.update_policy(policy_reward, policy_label, policy_sen_1, policy_sen_2, policy_vec_1, policy_vec_2)

            if fc % print_epoch == 0:
                print(m.validate(test_sen_1, test_sen_2, test_vec_1, test_vec_2, test_label))

    def re_init(self):
        self.session.close()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())


m = MatchModel(300, 10)
fx = np.transpose(np.load('fuck_x.npy'), (1, 0, 2))
fy = np.transpose(np.load('fuck_y.npy'), (1, 0, 2))
fl = np.load('fuck_label.npy')

tx = np.transpose(m.gen_seq(fx), (1, 0, 2))
ty = np.transpose(m.gen_seq(fy), (1, 0, 2))

for fsdfsd in range(10):
    start = time.time()
    m.train([fx], [fy], [tx], [ty], [fl], 10, fx, fy, tx, ty, fl, 1, 10)
    print(time.time() - start)
