'''
Adapted from the implementation of paper CCMI : Classifier based Conditional Mutual Information Estimation by Sudipto Mukherjee, Himanshu Asnani and Sreeram Kannan.
https://github.com/sudiptodip15/CCMI
'''

import random

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_slim as slim

eps = 1e-8

def plot_util(mi_with_iter, mi_true, mon_freq = 5000):
        plt.plot((np.arange(len(mi_with_iter)) + 1)*mon_freq, np.array(mi_with_iter))
        plt.axhline(y=mi_true, color='r')
        plt.xlabel('Iterations')
        plt.ylabel('I(X; Y)')
        plt.show()

def shuffle(batch_data, dx):

    batch_x = batch_data[:, 0:dx]
    batch_y = batch_data[:, dx:]
    batch_y = np.random.permutation(batch_y)
    return np.hstack((batch_x, batch_y))


def log_mean_exp_tf(fx_q, ax=0):

    max_ele = tf.reduce_max(fx_q, axis = ax, keepdims=True)
    return tf.squeeze(max_ele + tf.log(eps + tf.reduce_mean(tf.exp(fx_q-max_ele))))


def log_mean_exp_numpy(fx_q, ax = 0):

    max_ele = np.max(fx_q, axis=ax, keepdims = True)
    return (max_ele + np.log(eps + np.mean(np.exp(fx_q-max_ele), axis = ax, keepdims=True))).squeeze()

def smooth_ma(values, window_size=100):
    return [np.mean(values[i:i + window_size]) for i in range(0, len(values) - window_size)]

class Neural_MINE(object):

    def __init__(self, data_train, data_eval, dx, h_dim = 64, actv = tf.nn.relu, batch_size = 128,
                 optimizer = 'adam', lr = 0.0001, max_ep = 200, mon_freq = 5000, metric = 'f_divergence'):

        self.dim_x = dx
        self.data_dim = data_train.shape[1]
        self.X = data_train[:, 0:dx]
        self.Y = data_train[:, dx:]
        self.train_size = len(data_train)

        self.X_eval = data_eval[:, 0:dx]
        self.Y_eval = data_eval[:, dx:]
        self.eval_size = len(data_eval)

        # Hyper-parameters of statistical network
        self.h_dim = h_dim
        self.actv = actv

        # Hyper-parameters of training process
        self.batch_size = batch_size
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.optimizer = optimizer
        self.lr = lr
        self.mon_freq = mon_freq
        self.metric = metric
        self.tol = 1e-4

    def sample_p_finite(self, batch_size):
        index = np.random.randint(low = 0, high = self.train_size, size=batch_size)
        return np.hstack((self.X[index, :], self.Y[index, :]))

    def stat_net(self, inp, reuse=False):
        with tf.variable_scope('func_approx') as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = slim.fully_connected(inp, num_outputs=self.h_dim, activation_fn=self.actv,
                                      weights_initializer=tf.orthogonal_initializer)
            out = slim.fully_connected(fc1, num_outputs=1, activation_fn=tf.identity,
                                      weights_initializer=tf.orthogonal_initializer)
            return out

    def get_div(self, stat_inp_p, stat_inp_q):

        if self.metric == 'donsker_varadhan':
            return log_mean_exp_tf(stat_inp_q) - tf.reduce_mean(stat_inp_p)
        elif self.metric == 'f_divergence':
            return tf.reduce_mean(tf.exp(stat_inp_q - 1)) - tf.reduce_mean(stat_inp_p)
        else:
            raise NotImplementedError


    def train(self):

        # Define nodes for training process
        Inp_p = tf.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp_p')
        finp_p = self.stat_net(Inp_p)

        Inp_q = tf.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp_q')
        finp_q = self.stat_net(Inp_q, reuse=True)

        loss = self.get_div(finp_p, finp_q)
        mi_t = -loss

        if self.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss)
        elif self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1 = 0.5, beta2 = 0.999).minimize(loss)

        mi_with_iter = []
        #print('Estimating MI with metric = {}, opt = {}, lr = {}'.format(self.metric, self.optimizer, self.lr))
        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.Session(config = run_config) as sess:

            sess.run(tf.global_variables_initializer())

            eval_inp_p = np.hstack((self.X_eval, self.Y_eval))
            eval_inp_q = shuffle(eval_inp_p, self.dim_x)
            mi_est = -np.inf
            for it in range(self.max_iter):

                batch_inp_p = self.sample_p_finite(self.batch_size)
                batch_inp_q = shuffle(batch_inp_p, self.dim_x)

                MI, _ = sess.run([mi_t, opt], feed_dict={Inp_p: batch_inp_p, Inp_q: batch_inp_q})

                if ((it + 1) % self.mon_freq == 0 or (it + 1) == self.max_iter):

                    prev_est = mi_est
                    mi_est = sess.run(mi_t, feed_dict={Inp_p: eval_inp_p, Inp_q: eval_inp_q})
                    mi_with_iter.append(mi_est)

                    #print('Iter [%8d] : MI_est = %.4f' % (it + 1, mi_est))

                    if abs(prev_est - mi_est) < self.tol:
                        break

            # mi_with_iter = smooth_ma(mi_with_iter)
            # mi_est = mi_with_iter[-1]

            return mi_est, mi_with_iter

class Classifier_MI(object):

    def __init__(self, data_train, data_eval, dx, h_dim = 256, actv = tf.nn.relu, batch_size = 64,
                 optimizer='adam', lr=0.001, max_ep = 20, mon_freq = 5000,  metric = 'donsker_varadhan'):

        self.dim_x = dx
        self.data_dim = data_train.shape[1]
        self.X = data_train[:, 0:dx]
        self.Y = data_train[:, dx:]
        self.train_size = len(data_train)

        self.X_eval = data_eval[:, 0:dx]
        self.Y_eval = data_eval[:, dx:]
        self.eval_size = len(data_eval)

        # Hyper-parameters of statistical network
        self.h_dim = h_dim
        self.actv = actv

        # Hyper-parameters of training process
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.mon_freq = mon_freq
        self.metric = metric
        self.tol = 1e-4
        self.eps = 1e-8


        #Post Non-linear Cosine Data-set : h_dim = (256, 256), actv = tf.nn.relu, batch_size = 64, adam, lr = 0.001 (default momentum's), max_ep = 20, num_boot_iter = 10 
        #Flow-Cytometry Data-set : h_dim = (64, 64), actv = tf.nn.relu, batch_size = 64, adam, lr = 0.001 (default momentum's), max_ep = 10, num_boot_iter = 20.

        self.reg_coeff = 1e-3

    def sample_p_finite(self, batch_size):
        index = np.random.randint(low = 0, high = self.train_size, size=batch_size)
        return np.hstack((self.X[index, :], self.Y[index, :]))


    def classifier(self, inp, reuse = False):

        with tf.compat.v1.variable_scope('func_approx') as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = slim.fully_connected(inp, num_outputs = self.h_dim, activation_fn=self.actv,
                                      weights_regularizer=slim.l2_regularizer(self.reg_coeff))
            fc2 = slim.fully_connected(fc1, num_outputs = self.h_dim, activation_fn=self.actv,
                                      weights_regularizer=slim.l2_regularizer(self.reg_coeff))
            logit = slim.fully_connected(fc2, num_outputs = 1, activation_fn=None,
                                        weights_regularizer=slim.l2_regularizer(self.reg_coeff))
            prob = tf.nn.sigmoid(logit)

            return logit, prob

    def train_classifier_MLP(self):

        # Define tensorflow nodes for classifier
        Inp = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp')
        label = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1], name='label')

        logit, y_prob = self.classifier(Inp)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        l2_loss = tf.compat.v1.losses.get_regularization_loss()
        cost = tf.reduce_mean(cross_entropy) + l2_loss

        y_hat = tf.round(y_prob)
        correct_pred = tf.equal(y_hat, label)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if self.optimizer == 'sgd':
            opt_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)
        elif self.optimizer == 'adam':
            opt_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        #print('Training MLP classifier on Two-sample, opt = {}, lr = {}'.format(self.optimizer, self.lr))
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.33
        run_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config = run_config) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            eval_inp_p = np.hstack((self.X_eval, self.Y_eval))
            eval_inp_q = shuffle(eval_inp_p, self.dim_x)
            B = len(eval_inp_p)

            for it in range(self.max_iter):

                batch_inp_p = self.sample_p_finite(self.batch_size)
                batch_inp_q = shuffle(batch_inp_p, self.dim_x)

                batch_inp = np.vstack((batch_inp_p, batch_inp_q))
                by = np.vstack((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))
                batch_index = np.random.permutation(2*self.batch_size)
                batch_inp = batch_inp[batch_index]
                by = by[batch_index]

                L, _ = sess.run([cost, opt_step], feed_dict={Inp: batch_inp, label: by})

                if ((it + 1) % self.mon_freq == 0):

                    eval_inp = np.vstack((eval_inp_p, eval_inp_q))
                    eval_y = np.vstack((np.ones((B, 1)), np.zeros((B, 1))))
                    eval_acc = sess.run(accuracy, feed_dict={Inp: eval_inp, label: eval_y})
                    print('Iteraion = {}, Test accuracy = {}'.format(it+1, eval_acc))

            pos_label_pred_p = sess.run(y_prob, feed_dict={Inp: eval_inp_p})
            rn_est_p = (pos_label_pred_p+self.eps)/(1-pos_label_pred_p-self.eps)
            finp_p = np.log(np.abs(rn_est_p))

            pos_label_pred_q = sess.run(y_prob, feed_dict={Inp: eval_inp_q})
            rn_est_q = (pos_label_pred_q + self.eps) / (1 - pos_label_pred_q - self.eps)
            finp_q = np.log(np.abs(rn_est_q))

            #mi_est = np.mean(finp_p) - np.log(np.mean(np.exp(finp_q)))
            mi_est = np.mean(finp_p) - log_mean_exp_numpy(finp_q)

        return mi_est

class CCMI_Model(object):
    def __init__(self, X, Y, Z, tester, metric, num_boot_iter, h_dim, max_ep):

        self.dim_x = X.shape[1]
        self.dim_y = Y.shape[1]
        self.dim_z = Z.shape[1]
        self.data_xyz = np.hstack((X, Y, Z))
        self.data_xz = np.hstack((X, Z))
        self.threshold = 1e-4

        self.tester = tester
        self.metric = metric
        self.num_boot_iter = num_boot_iter
        self.h_dim = h_dim
        self.max_ep = max_ep

    def split_train_test(self, data):
        total_size = data.shape[0]
        train_size = int(2*total_size/3)
        data_train = data[0:train_size,:]
        data_test = data[train_size:, :]
        return data_train, data_test

    def gen_bootstrap(self, data):
        np.random.seed()
        random.seed()
        num_samp = data.shape[0]
        #I = np.random.choice(num_samp, size=num_samp, replace=True)
        I = np.random.permutation(num_samp)
        data_new = data[I, :]
        return data_new

    def get_cmi_est(self):

        if self.tester == 'Neural':
            print('Tester = {}, metric = {}'.format(self.tester, self.metric))
            if self.metric == 'donsker_varadhan':
                batch_size = 512
            else:
                batch_size = 128
            I_xyz_list = []
            for t in range(self.num_boot_iter):
                tf.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xyz)
                data_xyz_train, data_xyz_eval = self.split_train_test(data_t)
                neurMINE_xyz = Neural_MINE(data_xyz_train, data_xyz_eval, self.dim_x,
                                           metric= self.metric, batch_size = batch_size)
                I_xyz_t, _ = neurMINE_xyz.train()
                I_xyz_list.append(I_xyz_t)

            I_xyz_list = np.array(I_xyz_list)
            I_xyz = np.mean(I_xyz_list)

            I_xz_list = []
            for i in range(self.num_boot_iter):
                tf.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xz)
                data_xz_train, data_xz_eval = self.split_train_test(data_t)
                neurMINE_xz = Neural_MINE(data_xz_train, data_xz_eval, self.dim_x,
                                          metric= self.metric, batch_size = batch_size)
                I_xz_t, _ = neurMINE_xz.train()
                I_xz_list.append(I_xz_t)

            I_xz_list = np.array(I_xz_list)
            I_xz = np.mean(I_xz_list)
            cmi_est = I_xyz - I_xz

        elif self.tester == 'Classifier':
            print('Tester = {}, metric = {}'.format(self.tester, self.metric))
            I_xyz_list = []
            for t in range(self.num_boot_iter):
                tf.compat.v1.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xyz)
                data_xyz_train, data_xyz_eval = self.split_train_test(data_t)
                classMINE_xyz = Classifier_MI(data_xyz_train, data_xyz_eval, self.dim_x,
                                              h_dim = self.h_dim, max_ep = self.max_ep)
                I_xyz_t = classMINE_xyz.train_classifier_MLP()
                I_xyz_list.append(I_xyz_t)

            I_xyz_list = np.array(I_xyz_list)
            I_xyz = np.mean(I_xyz_list)

            I_xz_list = []
            for i in range(self.num_boot_iter):
                tf.compat.v1.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xz)
                data_xz_train, data_xz_eval = self.split_train_test(data_t)
                classMINE_xz = Classifier_MI(data_xz_train, data_xz_eval, self.dim_x,
                                              h_dim = self.h_dim, max_ep = self.max_ep)
                I_xz_t = classMINE_xz.train_classifier_MLP()
                I_xz_list.append(I_xz_t)

            I_xz_list = np.array(I_xz_list)
            I_xz = np.mean(I_xz_list)
            cmi_est = I_xyz - I_xz
        else:
            raise NotImplementedError

        return cmi_est

    def is_indp(self, cmi_est):
          if max(0, cmi_est) < self.threshold:
              return True
          else:
              return False

def CCMI(X, Y, Z, **kwargs):
    tf.compat.v1.disable_eager_execution()
    model_indp = CCMI_Model(X, Y, Z, tester = 'Classifier', metric = 'donsker_varadhan', num_boot_iter = 10, h_dim = 64, max_ep = 20)
    return model_indp.get_cmi_est()