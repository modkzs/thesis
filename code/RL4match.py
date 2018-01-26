# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class MatchModel(object):
    def __init__(self, word_vec_len, max_sen_len):
        # all model parameter
        self.max_sen_len = max_sen_len
        self.word_evc_len = word_vec_len
        self.actions = [[0, 1], [1, 0], [1, 1]]

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

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def deep_match_model(self, lr):
        with tf.variable_scope('deep_match_model'):
            input_1 = tf.placeholder(shape=[None, self.max_sen_len, self.word_evc_len], dtype=tf.float32, name="q1")
            input_2 = tf.placeholder(shape=[None, self.max_sen_len, self.word_evc_len], dtype=tf.float32, name="q2")
            label = tf.placeholder(shape=[None, 1], dtype=np.float32, name="label")

            inputs_1 = tf.unstack(input_1, self.max_sen_len, 1)
            inputs_2 = tf.unstack(input_2, self.max_sen_len, 1)

            outputs_1, states_1 = rnn.static_rnn(
                tf.nn.rnn_cell.LSTMCell(self.deep_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE,
                                        initializer=tf.orthogonal_initializer(dtype=tf.float32)), inputs_1,
                dtype=tf.float32)
            outputs_2, states_2 = rnn.static_rnn(
                tf.nn.rnn_cell.LSTMCell(self.deep_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE,
                                        initializer=tf.orthogonal_initializer(dtype=tf.float32)), inputs_2,
                dtype=tf.float32)
            final_output = tf.concat([outputs_1[-1], outputs_2[-1]], axis=1)

            weight = tf.Variable(tf.random_normal([self.deep_hidden_num * 2, 1], dtype=np.float32), dtype=np.float32)
            bias = tf.Variable(tf.random_normal([1, 1], dtype=np.float32), dtype=np.float32)
            logits = tf.matmul(final_output, weight) + bias
            prediction = tf.nn.sigmoid(logits)

            loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss_op)

            correct_pred = tf.equal(tf.cast(tf.greater(prediction, 0.5), tf.float32), label)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            return {"outputs_1": outputs_1, "outputs_2": outputs_2, "train_op": train_op,
                    "accuracy": accuracy, "input_1": input_1, "input_2": input_2, "label": label}

    def train_deep_model(self, sen_1, sen_2, label, batch_size):
        train_sen_1 = sen_1[:int(len(sen_1) * 0.8)]
        train_sen_2 = sen_2[:int(len(sen_1) * 0.8)]
        train_y = label[:int(len(sen_1) * 0.8)]

        dev_sen_1 = sen_1[int(len(sen_1) * 0.8):int(len(sen_1) * 0.9)]
        dev_sen_2 = sen_2[int(len(sen_1) * 0.8):int(len(sen_1) * 0.9)]
        dev_y = np.array(label[int(len(sen_1) * 0.8):int(len(sen_1) * 0.9)]).reshape([-1, 1])

        test_sen_1 = sen_1[int(len(sen_1) * 0.9):]
        test_sen_2 = sen_2[int(len(sen_1) * 0.9):]
        test_y = np.array(label[int(len(sen_1) * 0.9):]).reshape([-1, 1])

        # used for shuffle
        seq = list(range(len(train_sen_1)))

        for k in range(20):
            np.random.shuffle(seq)

            for i in range(0, len(train_sen_1), batch_size):
                tmp_x_1 = np.array([train_sen_1[j] for j in seq[i:i + batch_size + 1]])
                tmp_x_2 = np.array([train_sen_2[j] for j in seq[i:i + batch_size + 1]])
                tmp_y = np.array([train_y[j] for j in seq[i:i + batch_size + 1]]).reshape([-1, 1])
                train_acc = self.session.run([self.deep_model['train_op'], self.deep_model['accuracy']],
                                             feed_dict={self.deep_model['input_1']: tmp_x_1,
                                                        self.deep_model['input_2']: tmp_x_2,
                                                        self.deep_model['label']: tmp_y})[-1]
                print([train_acc, self.session.run(self.deep_model['accuracy'],
                                                   feed_dict={self.deep_model['input_1']: dev_sen_1,
                                                              self.deep_model['input_2']: dev_sen_2,
                                                              self.deep_model['label']: dev_y})[0]]
                      , end=',')

            print('\ntest accuracy:', self.session.run(self.deep_model['accuracy'],
                                                       feed_dict={self.deep_model['input_1']: test_sen_1,
                                                                  self.deep_model['input_2']: test_sen_2,
                                                                  self.deep_model['label']: test_y}))

    def gen_seq(self, sen_vec):
        return self.session.run(self.deep_model['outputs_1'], feed_dict={self.deep_model['input_1']: sen_vec})

    def policy_model(self, lr):
        with tf.variable_scope('policy_model'):
            sen_1 = tf.placeholder(shape=[None, self.deep_hidden_num], dtype=np.float32, name="sen1")
            sen_2 = tf.placeholder(shape=[None, self.deep_hidden_num], dtype=np.float32, name="sen2")

            vec_1 = tf.placeholder(shape=[None, self.word_evc_len], dtype=np.float32, name="vec1")
            vec_2 = tf.placeholder(shape=[None, self.word_evc_len], dtype=np.float32, name="vec2")

            # TODO: is concat too simple?
            final_vec = tf.concat([sen_1, sen_2, vec_1, vec_2], axis=1)

            weight = tf.Variable(
                tf.random_normal([(self.word_evc_len + self.deep_hidden_num) * 2, 3], dtype=np.float32),
                dtype=np.float32)
            bias = tf.Variable(tf.random_normal([3, ], dtype=np.float32), dtype=np.float32)
            logits = tf.matmul(final_vec, weight) + bias
            prediction = tf.nn.softmax(logits)

            learning_rate = tf.constant([lr], dtype=np.float32, shape=[])
            # TODO: can not run in batch? WTF
            gradient = tf.placeholder(shape=[], dtype=np.float32)
            lr_node = tf.multiply(learning_rate, gradient)

            label = tf.placeholder(shape=[None, 3], dtype=np.float32, name='label')

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_node)
            train_op = optimizer.minimize(loss_op)

            return {'sen_1': sen_1, 'sen_2': sen_2, 'vec_1': vec_1, 'vec_2': vec_2, 'label': label,
                    'prediction': prediction, 'optimizer': train_op, 'reward': gradient}

    # hidden of sen_1 and sen_2 from DeepMatch model, and 2 word vector    
    def get_policy(self, sen_1, sen_2, vec_1, vec_2):
        return self.session.run(self.policy['prediction'],
                                feed_dict={self.policy['sen_1']: sen_1, self.policy['sen_2']: sen_2,
                                           self.policy['vec_1']: vec_1, self.policy['vec_2']: vec_2})

    # how to update?
    def update_policy(self, reward, seq, sen_1, sen_2, maps_1, maps_2):
        vec_1 = self.gen_seq(np.array(sen_1).reshape(1, 100, 300))
        vec_2 = self.gen_seq(np.array(sen_2).reshape(1, 100, 300))

        # TODO: vectorize it to speed up or update in GPU?
        i = -1
        for p in seq:
            self.session.run(self.policy['optimizer'], {self.policy['reward']: reward[0],
                                                        # TODO: refactor variable name.......
                                                        self.policy['vec_1']: sen_1[maps_1[i]].reshape(1, -1),
                                                        self.policy['vec_2']: sen_2[maps_2[i]].reshape(1, -1),
                                                        self.policy['sen_1']: vec_1[maps_1[i]].reshape(1, -1),
                                                        self.policy['sen_2']: vec_2[maps_2[i]].reshape(1, -1),
                                                        self.policy['label']: p.reshape(1, -1)})
            i -= 1

    # TODO: This train is slow, need to rewrite it, the execute cell is 3 LSTM cell not a LSTM sentence
    def reward_model(self, lr):
        with tf.variable_scope('reward_model'):
            # sentence input
            sen = tf.placeholder(shape=[None, self.max_sen_len, self.word_evc_len], dtype=np.float32, name="q1")
            # action seq input, each input is 3d, corresponding to 3 prob
            seqs = tf.placeholder(shape=[None, self.max_sen_len * 2, 3], dtype=np.float32, name="seq")
            # final label
            label = tf.placeholder(shape=[None, 1], dtype=np.float32, name="label")

            token = tf.unstack(sen, self.max_sen_len, 1)

        # TODO: using this RNN is very slow. How good? may other choice(like using pre-train RNN or just word_vec)?
        with tf.variable_scope('reward_model_sen'):
            # using LSTM to encode input sentence, output should be input of seq model
            outputs, _ = rnn.static_rnn(
                tf.nn.rnn_cell.BasicLSTMCell(self.reward_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE), token,
                dtype=tf.float32)

            # mean is useless when predict, only used to compute gradient
            outputs = tf.stack(outputs)
            mean = tf.reduce_sum(outputs)

            learning_rate = tf.constant([lr], shape=[])
            gradient = tf.placeholder(shape=[], dtype=np.float32)
            lr_node = tf.multiply(learning_rate, gradient)

            optimizer_sen = tf.train.GradientDescentOptimizer(learning_rate=lr_node)
            train_op_sen = optimizer_sen.minimize(mean)

        with tf.variable_scope('reward_model_seq'):
            # encode seq, the final output
            seq_sen_1 = tf.placeholder(shape=[None, 2 * self.max_sen_len, self.reward_hidden_num], dtype=np.float32,
                                       name="q1")
            seq_sen_2 = tf.placeholder(shape=[None, 2 * self.max_sen_len, self.reward_hidden_num], dtype=np.float32,
                                       name="q2")

            seq_sen_var_1 = tf.Variable(np.float32, validate_shape=False,
                                        expected_shape=[None, 2 * self.max_sen_len, self.reward_hidden_num])
            seq_sen_var_2 = tf.Variable(np.float32, validate_shape=False,
                                        expected_shape=[None, 2 * self.max_sen_len, self.reward_hidden_num])

            seq_sen_var_1 = tf.assign(seq_sen_var_1, seq_sen_1, validate_shape=False)
            seq_sen_var_2 = tf.assign(seq_sen_var_2, seq_sen_2, validate_shape=False)

            # TODO: concat?
            seq_input = tf.concat([seq_sen_var_1, seq_sen_var_2, seqs], axis=2)

            seq_input = tf.unstack(seq_input, self.max_sen_len * 2, 1)

            outputs_seq, _ = rnn.static_rnn(
                tf.nn.rnn_cell.BasicLSTMCell(self.reward_hidden_num, state_is_tuple=True, reuse=tf.AUTO_REUSE),
                seq_input, dtype=tf.float32)
            weight = tf.Variable(tf.random_normal([self.reward_hidden_num, 1], dtype=np.float32), dtype=np.float32)
            bias = tf.Variable(tf.random_normal([1, 1], dtype=np.float32), dtype=np.float32)
            logits = tf.matmul(outputs_seq[-1], weight) + bias
            prediction = tf.nn.sigmoid(logits)

            loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label))
            optimizer_seq = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_op_seq = optimizer_seq.minimize(loss_op)

            gradient_1 = tf.reduce_mean(tf.gradients(loss_op, seq_sen_var_1))
            gradient_2 = tf.reduce_mean(tf.gradients(loss_op, seq_sen_var_2))

        return {'sen': sen, 'seqs': seqs, 'outputs': outputs, 'gradient': gradient, 'prediction': prediction,
                'train_op_seq': train_op_seq, 'train_op_sen': train_op_sen,
                'seq_sen_1': seq_sen_1, 'seq_sen_2': seq_sen_2, 'label': label,
                'sen_gradient_1': gradient_1, 'sen_gradient_2': gradient_2}

    def get_reward(self, sen_1, sen_2, seqs, map_1, map_2):
        output_1 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_1})
        output_2 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_2})

        output_1 = np.array([output_1[i].reshape(-1) for i in map_1]).reshape(1, 200, 128)
        output_2 = np.array([output_2[i].reshape(-1) for i in map_2]).reshape(1, 200, 128)

        return self.session.run(self.reward['prediction'],
                                feed_dict={self.reward['seq_sen_1']: output_1, self.reward['seq_sen_2']: output_2,
                                           self.reward['seqs']: seqs})

    # a really fucking complicated function. 
    # Must compte reward seq model first, then get each gradient to reward sentence model, then update sentence model
    def update_reward(self, sen_1, sen_2, seqs, map_1, map_2, labels):
        output_1 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_1})
        output_2 = self.session.run(self.reward['outputs'], feed_dict={self.reward['sen']: sen_2})

        output_1 = np.array([output_1[i].reshape(-1) for i in map_1]).reshape(1, 200, 128)
        output_2 = np.array([output_2[i].reshape(-1) for i in map_2]).reshape(1, 200, 128)

        grad = self.session.run(
            [self.reward['train_op_seq'], self.reward['sen_gradient_1'], self.reward['sen_gradient_2']],
            feed_dict={self.reward['seq_sen_1']: output_1, self.reward['seq_sen_2']: output_2,
                       self.reward['seqs']: seqs, self.reward['label']: labels}
        )[1:]

        # TODO: each cell get same weight, actually should use map to get cell use weight and using it update model?
        self.session.run(self.reward['train_op_sen'],
                         feed_dict={self.reward['sen']: sen_1, self.reward['gradient']: grad[0]})
        self.session.run(self.reward['train_op_sen'],
                         feed_dict={self.reward['sen']: sen_2, self.reward['gradient']: grad[1]})

    def fill(self, vec, length):
        return np.array(vec + [np.array([0] * self.word_evc_len)] * (length - len(vec)))

    def monte_carlo(self, sen_1, sen_2, word_vec, gen_num):
        len_1 = len(sen_1)
        len_2 = len(sen_2)

        vec_1 = [word_vec[s] for s in sen_1 if s in word_vec]
        vec_2 = [word_vec[s] for s in sen_2 if s in word_vec]

        vec_1 = np.array(self.fill(vec_1, self.max_sen_len)).reshape(1, self.max_sen_len, self.word_evc_len)
        vec_2 = np.array(self.fill(vec_2, self.max_sen_len)).reshape(1, self.max_sen_len, self.word_evc_len)

        # TODO: we currently use a pre-trained model to represent sentence, may other way(like using reward RNN)?
        vec_1 = self.gen_seq(vec_1)
        vec_2 = self.gen_seq(vec_2)

        pos = [0, 0]

        directions = []
        maps_1 = []
        maps_2 = []
        labels = []

        for i in range(gen_num):
            direction = []
            map_1 = []
            map_2 = []

            while pos[0] < len_1 and pos[1] < len_2:
                map_1.append(pos[0])
                map_2.append(pos[1])

                p = self.get_policy(vec_1[pos[0]], vec_2[pos[1]], word_vec[sen_1[pos[0]]].reshape(1, -1),
                                    word_vec[sen_2[pos[1]]].reshape(1, -1))[0]

                if p[0] > p[1] and p[0] > p[2]:
                    action = 0
                    labels.append([1, 0, 0])
                elif p[1] > p[0] and p[1] > p[2]:
                    action = 1
                    labels.append([0, 1, 0])
                else:
                    action = 2
                    labels.append([0, 0, 1])

                pos[0] += self.actions[action][0]
                pos[1] += self.actions[action][1]

                direction.append(p)

            if pos[0] == len_1:
                direction += np.array([1, 0, 0]) * (len_2 - pos[1])
                labels += [[1, 0, 0]] * (len_2 - pos[1])
                map_1 += [pos[0]] * (len_2 - pos[1])
                map_2 += [i for i in range(pos[1], len_2)]
            else:
                direction += np.array([0, 1, 0]) * (len_1 - pos[0])
                map_1 += [i for i in range(pos[0], len_1)]
                map_2 += [pos[1]] * (len_1 - pos[0])
                labels += [[0, 1, 0]] * (len_2 - pos[1])

            maps_1.append(map_1)
            maps_2.append(map_2)
            directions.append(direction)

        return maps_1, maps_2, directions, labels

    def predict(self, sen_1, sen_2, word_vec):
        sens_1 = [s.split(' ') for s in sen_1]
        sens_2 = [s.split(' ') for s in sen_2]

        sen_vec_1 = [[word_vec[s] for s in sen] for sen in sens_1]
        sen_vec_2 = [[word_vec[s] for s in sen] for sen in sens_2]

        sen_vec_1 = [s + [[0] * self.word_evc_len] * (self.max_sen_len - len(s)) for s in sen_vec_1]
        sen_vec_2 = [s + [[0] * self.word_evc_len] * (self.max_sen_len - len(s)) for s in sen_vec_2]

        tmps = self.monte_carlo(sens_1, sens_2, word_vec, 1)

        return self.get_reward(sen_vec_1, sen_vec_2, tmps[2], tmps[0], tmps[1])

    # TODO: wait to write.....
    def validate(self, sen_1, sen_2, label):
        pass

    # TODO: word_vec is so fucking huge, how to kick it out?
    def train(self, sens_1, sens_2, labels, batch_size, word_vec):
        sens_1 = [s.split(' ') for s in sens_1]
        sens_2 = [s.split(' ') for s in sens_2]

        sen_vec_1 = [[word_vec[s] for s in sen] for sen in sens_1]
        sen_vec_2 = [[word_vec[s] for s in sen] for sen in sens_2]

        # first train deep model
        sen_vec_1 = [s + [np.array([0] * self.word_evc_len)] * (self.max_sen_len - len(s)) for s in sen_vec_1]
        sen_vec_2 = [s + [np.array([0] * self.word_evc_len)] * (self.max_sen_len - len(s)) for s in sen_vec_2]

        self.train_deep_model(sen_vec_1, sen_vec_2, labels, batch_size)

        # iterator nums
        for fc in range(1):
            for j in range(len(sens_1)):
                map_1, map_2, directions, label = self.monte_carlo(sens_1[j], sens_2[j], word_vec, 1)
                vec_1 = sen_vec_1[j]
                vec_2 = sen_vec_2[j]

                # TODO: should be much simulation
                directions = directions[0]
                map_1 = map_1[0]
                map_2 = map_2[0]

                # TODO: delete it when switch to dynamic_rnn?
                map_1 = map_1 + [map_1[-1]] * (2 * self.max_sen_len - len(map_1))
                map_2 = map_2 + [map_2[-1]] * (2 * self.max_sen_len - len(map_2))
                directions = np.array(
                    list(directions) + [directions[-1]] * (2 * self.max_sen_len - len(directions))).reshape([1, 200, 3])

                vec_1 = np.array(vec_1 + np.array([0] * self.word_evc_len) * (self.max_sen_len - len(vec_1))).reshape(
                    [1, 100, 300])
                vec_2 = np.array(vec_2 + np.array([0] * self.word_evc_len) * (self.max_sen_len - len(vec_2))).reshape(
                    [1, 100, 300])

                # TODO:update one batch or whole data set?
                self.update_reward(vec_1, vec_2, directions, map_1, map_2, np.array(labels).reshape([1, 1]))

                reward = self.get_reward(vec_1, vec_2, directions, map_1, map_2)

                # TODO: this as vectorize?
                for i in range(len(directions)):
                    self.update_policy(reward[i], directions[i], sen_vec_1[i], sen_vec_2[i], map_1, map_2)


m = MatchModel(300, 100)
xs_1 = ['What', 'is', 'the', 'step', 'by', 'step', 'guide', 'to', 'invest', 'in', 'share', 'market', 'in', 'india', '?']
xs_2 = ['What', 'is', 'the', 'step', 'by', 'step', 'guide', 'to', 'invest', 'in', 'share', 'market', '?']

word2vec = {}
for w in set(xs_1 + xs_2):
    word2vec[w] = np.random.rand(300, )

m.train([" ".join(xs_1)], [" ".join(xs_2)], [1], 1, word2vec)
