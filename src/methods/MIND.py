"""
TensorFlow Implementation of MIND ([1]) under Spearman rank correlation constraints.
[1] Kom Samo, Y. (2021). Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach. Proceedings of The 24th International Conference on Artificial Intelligence and Statistics. Available from https://proceedings.mlr.press/v130/kom-samo21a.html.
https://github.com/kxytechnologies/kxy-python
"""

import numpy as np
import logging

import tensorflow as tf
from tensorflow import keras
keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.optimizers import Adam
from keras.utils import Sequence
from keras import Model
from keras.backend import clip
from keras.layers import Dense, Lambda, concatenate, Dot
from keras.initializers import GlorotUniform
from tensorflow.python.ops import math_ops
from keras.layers import Layer
from keras.losses import Loss

rankdata = lambda x: 1.+np.argsort(np.argsort(x, axis=0), axis=0)
LOCAL_SEED = 0
INITIALIZER_COUNT = 0

"""
Global default training configs
"""
# LEARNING PARAMETERS
LR = 0.005
EPOCHS = 20

# ADAM PARAMETERS
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-04
AMSGRAD = False
BATCH_SIZE = 500


def set_default_parameter(name, value):
	'''
	Utility function to change parameters above at runtime.
	'''
	import logging
	globals()[name.upper()] = value
	return

def get_default_parameter(name):
	return eval(name.upper())


class CopulaBatchGenerator(Sequence):
	''' 
	Random batch generator of maximum-entropy copula learning.
	'''
	def __init__(self, z, batch_size=1000, steps_per_epoch=100):
		self.batch_size = batch_size
		self.d = z.shape[1]
		self.n = z.shape[0]
		self.z = z
		self.steps_per_epoch = steps_per_epoch
		self.emp_u = rankdata(self.z)/(self.n + 1.)
		self.emp_u[np.isnan(self.z)] = 0.5
		self.rnd_gen = np.random.default_rng(LOCAL_SEED)

		if self.n < 200*self.d:
			dn = 200*self.d - self.n
			selected_rows = self.rnd_gen.choice(self.n, dn, replace=True)
			emp_u = self.emp_u[selected_rows, :].copy()
			scale = 1./(100.*self.n)
			emp_u += (scale*self.rnd_gen.uniform(size=emp_u.shape) - 0.5*scale)
			self.emp_u = np.concatenate([self.emp_u, emp_u], axis=0)
			self.n = self.emp_u.shape[0]

		self.batch_selector = self.rnd_gen.choice(self.n, self.batch_size*self.steps_per_epoch, replace=True)
		self.batch_selector = self.batch_selector.reshape((self.steps_per_epoch, self.batch_size))


	def getitem_ndarray(self, idx):
		''' '''
		i = idx % self.steps_per_epoch
		selected_rows = self.batch_selector[i]
		emp_u_ = self.emp_u[selected_rows, :]
		z_p = emp_u_.copy()
		z_q = self.rnd_gen.uniform(size=emp_u_.shape)

		z = np.empty((self.batch_size, self.d, 2))
		z[:, :, 0] = z_p
		z[:, :, 1] = z_q
		batch_x = z
		batch_y = np.ones((self.batch_size, 2))  # Not used  
		return batch_x, batch_y


	def __getitem__(self, idx):
		''' '''
		batch_x, batch_y = self.getitem_ndarray(idx)
		batch_x, batch_y = tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)
		return batch_x, batch_y


	def __len__(self):
		return self.steps_per_epoch

def frozen_glorot_uniform():
	'''
	Deterministic GlorotUniform initializer.
	'''
	if LOCAL_SEED is not None:
		initializer =  GlorotUniform(LOCAL_SEED+INITIALIZER_COUNT)
		globals()['INITIALIZER_COUNT'] = INITIALIZER_COUNT + 1
		return initializer
	else:
		return GlorotUniform()

class InitializableDense(Layer):
	''' 
	'''
	def __init__(self, units, initial_w=None, initial_b=None, bias=False):
		'''
		initial_w should be None or a 2D numpy array.
		initial_b should be None or a 1D numpy array.
		'''
		super(InitializableDense, self).__init__()
		self.units = units
		self.with_bias = bias
		self.w_initializer = 'zeros' if initial_w is None else tf.constant_initializer(initial_w)

		if self.with_bias:
			self.b_initializer = 'zeros' if initial_b is None else tf.constant_initializer(initial_b)


	def build(self, input_shape):
		''' '''
		self.w = self.add_weight(shape=(input_shape[-1], self.units), \
			initializer=self.w_initializer, trainable=True, name='quad_w')

		if self.with_bias:
			self.b = self.add_weight(shape=(self.units,), \
				initializer=self.b_initializer, trainable=True, name='quad_b')


	def call(self, inputs):
		''' '''
		return tf.matmul(inputs, self.w)+self.b if self.with_bias else tf.matmul(inputs, self.w)

class CopulaModel(Model):
	"""
	Maximum-entropy copula under (possibly sparse) Spearman rank correlation constraints.
	"""
	def __init__(self, d, subsets=[]):
		super(CopulaModel, self).__init__()
		self.d = d
		if subsets == []:
			subsets = [[_ for _ in range(d)]]

		self.subsets = subsets
		self.n_subsets = len(self.subsets)
		self.p_samples = Lambda(lambda x: x[:,:,0])
		self.q_samples = Lambda(lambda x: x[:,:,1])

		self.fx_non_mon_layer_1s = [Dense(3, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform()) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_2s = [Dense(5, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform()) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_3s = [Dense(3, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform()) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_4s = [Dense(1) for _ in range(self.n_subsets)]

		eff_ds = [len(subset)+1 for subset in self.subsets]
		self.spears = [InitializableDense(eff_d) for eff_d in eff_ds]
		self.dots = [Dot(1) for _ in range(self.n_subsets)]

		# Mixing layers
		self.mixing_layer1 = Dense(5, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.mixing_layer2 = Dense(5, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.mixing_layer3 = Dense(1, kernel_initializer=frozen_glorot_uniform())

	def subset_statistics(self, u, i):
		'''
		Statistics function for the i-th subset of variables.
		'''
		n = tf.shape(u)[0]
		ui = tf.gather(u, self.subsets[i], axis=1)
		res = tf.zeros(shape=[n, 1], dtype=tf.float64)

		# Constraints beyond quadratic
		fui = self.fx_non_mon_layer_1s[i](ui)
		fui = self.fx_non_mon_layer_2s[i](fui)
		fui = self.fx_non_mon_layer_3s[i](fui)
		fui = self.fx_non_mon_layer_4s[i](fui)
		ui = concatenate([ui, fui], axis=1)
	
		# Spearman terms
		spearman_term = self.spears[i](ui)
		spearman_term = self.dots[i]([spearman_term, ui])
		res = tf.add(res, spearman_term)
		return res


	def statistics(self, u):
		'''
		Statistics function.
		''' 
		if self.n_subsets > 1:
			ts = [self.subset_statistics(u, i) for i in range(self.n_subsets)]
			t = concatenate(ts, axis=1)
			t = self.mixing_layer1(t)
			t = self.mixing_layer2(t)
			t = self.mixing_layer3(t)
		else:
			t = self.subset_statistics(u, 0)
		return t


	def call(self, inputs):
		'''
		'''
		p_samples = self.p_samples(inputs)
		t_p = self.statistics(p_samples)

		q_samples = self.q_samples(inputs)
		t_q = self.statistics(q_samples)
		
		t = concatenate([t_p, t_q], axis=1)
		t = clip(t, -100., 100.)
		return t


	def copula(self, inputs):
		'''
		'''
		u = tf.constant(inputs)
		c = math_ops.exp(self.statistics(u))
		return c.numpy()/c.numpy().mean()

class MINDLoss(Loss):  
	'''
	MIND loss function: :math:`-E_P(T(x, y)^T\theta) + \log E_Q(e^{T(x, y)^T\theta})`.
	'''
	def call(self, y_true, y_pred):
		''' '''
		p_samples = y_pred[:, 0]
		q_samples = y_pred[:, 1]
		mi = -tf.reduce_mean(p_samples) + math_ops.log(tf.reduce_mean(math_ops.exp(q_samples)))
		return mi


class CopulaLearner(object):
	'''
	Maximum-entropy learner.
	'''
	def __init__(self, d, beta_1=None, beta_2=None, epsilon=None, amsgrad=None, \
			name='Adam', lr=None, subsets=[]):
		self.d = d
		self.model = CopulaModel(self.d, subsets=subsets)
		beta_1 = get_default_parameter('beta_1') if beta_1 is None else beta_1
		beta_2 = get_default_parameter('beta_2') if beta_2 is None else beta_2
		lr = get_default_parameter('lr') if lr is None else lr
		amsgrad = get_default_parameter('amsgrad') if amsgrad is None else amsgrad
		epsilon = get_default_parameter('epsilon') if epsilon is None else epsilon
		logging.info('Using the Adam optimizer with learning parameters: ' \
			'lr: %.4f, beta_1: %.4f, beta_2: %.4f, epsilon: %.8f, amsgrad: %s' % \
			(lr, beta_1, beta_2, epsilon, amsgrad))
		self.opt = Adam(beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, \
			name=name, lr=lr)
		self.loss = MINDLoss()
		assert tf.executing_eagerly()
		self.model.compile(optimizer=self.opt, loss=self.loss, run_eagerly=True)
		self.copula_entropy = None


	def fit(self, z, batch_size=64, steps_per_epoch=1000, epochs=None):
		''' '''
		z_gen = CopulaBatchGenerator(z, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
		epochs = get_default_parameter('epochs') if epochs is None else epochs
		self.model.fit(z_gen, epochs=epochs, batch_size=None, steps_per_epoch=steps_per_epoch, \
			callbacks=[EarlyStopping(patience=3, monitor='loss'), TerminateOnNaN()], verbose=0)
		self.copula_entropy = self.model.evaluate(z_gen)

def copula_entropy(z, subsets=[]):
	'''
	Estimate the entropy of the copula distribution of a d-dimensional random vector using MIND ([1]) with Spearman rank correlation constraints.
	Parameters
	----------
	z : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.
	Returns
	-------
	ent : float
		The estimated copula entropy.
	'''
	if len(z.shape)==1 or z.shape[1]==1:
		return 0.0

	d = z.shape[1]
	
	cl = CopulaLearner(d, subsets=subsets)
	cl.fit(z)
	ent = min(cl.copula_entropy, 0.0)

	return ent

def MIND(X, Y, Z=None, **kwargs):
	tf.compat.v1.enable_eager_execution()
	X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
	# X, Y, Z = map(strip_outliers, (X, Y, Z))
	y, x = X, Y
	'''
	Estimate the mutual information between two random vectors using MIND ([1]) with Spearman rank correlation constraints.
	Parameters
	----------
	y : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.
	x : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.
	Returns
	-------
	mi : float
		The estimated mutual information.
	'''
	y = y[:, None] if len(y.shape)==1 else y
	x = x[:, None] if len(x.shape)==1 else x
	z = np.concatenate([y, x], axis=1)
	huy = copula_entropy(y)
	hux = copula_entropy(x)
	huz = copula_entropy(z)
	mi = max(huy+hux-huz, 0.0)

	return mi