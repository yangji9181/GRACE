from NN import *

class GRACE(object):
	def __init__(self, paras, graph):
		self.paras = paras
		self.build(graph)

	def build(self, graph):
		self.build_variable(graph)
		self.build_loss()

	def build_variable(self, graph):
		self.training = tf.placeholder(tf.bool)
		self.X = tf.Variable(graph.feature, trainable=False, dtype=tf.float32)
		dense_shape = [self.paras.num_node, self.paras.num_node]
		# random walk outgoing
		self.T = tf.SparseTensor(indices=graph.indices, values=graph.T_values, dense_shape=dense_shape)
		# influence propagation
		self.RI = tf.Variable(graph.RI, trainable=False, dtype=tf.float32)
		# random walk propagation
		self.RW = tf.Variable(graph.RW, trainable=False, dtype=tf.float32)
		self.mean = weight('mean', [self.paras.num_cluster, self.paras.embed_dim])
		self.P = tf.placeholder(tf.float32, [self.paras.num_node, self.paras.num_cluster])
		self.Z = self.encode()
		self.Z_transform = self.transform()
		self.Q = self.build_Q()

	def build_loss(self):
		X_p = self.decode()
		self.loss_r, self.loss_c = self.build_loss_r(X_p), self.build_loss_c()
		pre_loss = self.loss_r
		pre_optimizer = getattr(tf.train, self.paras.optimizer + 'Optimizer')(learning_rate=self.paras.learning_rate)
		self.pre_gradient_descent = pre_optimizer.minimize(pre_loss)
		loss = self.paras.lambda_r * self.loss_r + self.paras.lambda_c * self.loss_c
		optimizer = getattr(tf.train, self.paras.optimizer + 'Optimizer')(learning_rate=self.paras.learning_rate)
		self.gradient_descent = optimizer.minimize(loss)

	def build_Q(self):
		Z = self.Z_transform
		Z = tf.tile(tf.expand_dims(Z, 1), tf.stack([1, self.paras.num_cluster, 1]))
		Q = tf.pow(tf.reduce_sum(tf.squared_difference(Z, self.mean), axis=2) / self.paras.epsilon + 1.0, -(self.paras.epsilon + 1.0) / 2.0)
		return Q / tf.reduce_sum(Q, axis=1, keep_dims=True)

	def build_loss_r(self, X_p):
		# todo: check this
		return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=X_p), axis=1))

	def build_loss_c(self):
		loss_c = tf.reduce_mean(self.P * tf.log(self.P / self.Q))
		loss_c = tf.verify_tensor_all_finite(loss_c, 'check nan')
		return loss_c

	def transform(self):
		transition_function = self.paras.transition_function
		Z = self.Z
		if transition_function == 'T':
			for i in range(self.paras.random_walk_step):
				Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), Z)
		elif transition_function in ['RI', 'RW']:
			Z = tf.matmul(self.__getattribute__(transition_function), Z, transpose_a=True)
		else:
			raise ValueError('Invalid transition function')
		if self.paras.BN:
			Z = batch_normalization(Z, 'Z')
		return Z

	def encode(self):
		hidden = self.X
		for i, dim in enumerate(self.paras.encoder_hidden + [self.paras.embed_dim]):
			hidden = fully_connected(hidden, dim, 'encoder_' + str(i))
			hidden = dropout(hidden, self.paras.keep_prob, self.training)
		return hidden

	def decode(self):
		hidden = self.Z
		for i, dim in enumerate(self.paras.decoder_hidden):
			hidden = fully_connected(hidden, dim, 'decoder_' + str(i))
			hidden = dropout(hidden, self.paras.keep_prob, self.training)
		return fully_connected(hidden, self.paras.feat_dim, 'decoder_' + str(len(self.paras.decoder_hidden)), activation='linear')

	def get_embedding(self, sess):
		return sess.run(self.Z_transform, feed_dict={self.training: False})

	def init_mean(self, mean, sess):
		sess.run(self.mean.assign(mean))

	def get_P(self, sess):
		P = tf.square(self.Q) / tf.reduce_sum(self.Q, axis=0)
		return sess.run(P / tf.reduce_sum(P, axis=1, keep_dims=True), feed_dict={self.training: False})

	def predict(self, sess):
		return sess.run(tf.one_hot(tf.argmax(self.Q, axis=1), depth=self.paras.num_cluster, on_value=1, off_value=0), feed_dict={self.training: False})



class GRACE_Dense(GRACE):
	def build_variable(self, graph):
		self.training = tf.placeholder(tf.bool)
		self.X = tf.Variable(graph.feature, trainable=False, dtype=tf.float32)
		# influence propagation
		self.RI = tf.placeholder(tf.float32, [self.paras.num_node, None])
		# random walk propagation
		self.RW = tf.placeholder(tf.float32, [self.paras.num_node, None])
		self.mean = weight('mean', [self.paras.num_cluster, self.paras.embed_dim])
		self.P = tf.placeholder(tf.float32, [None, self.paras.num_cluster])
		self.Z = self.encode()
		self.Z_transform = self.transform()
		self.Q = self.build_Q()

	def build_loss_c(self):
		loss_c = tf.reduce_mean(self.P * tf.log(self.P / self.Q))
		loss_c = tf.verify_tensor_all_finite(loss_c, 'check nan')
		return loss_c

	def transform(self):
		transition_function = self.paras.transition_function
		Z = self.Z
		if transition_function in ['RI', 'RW']:
			Z = tf.matmul(self.__getattribute__(transition_function), Z, transpose_a=True)
		else:
			raise ValueError('Invalid transition function')
		if self.paras.BN:
			Z = batch_normalization(Z, 'Z')
		return Z

	def get_P(self, sess):
		raise NotImplementedError

	def get_embedding(self, sess):
		raise NotImplementedError

	def predict(self, sess):
		raise NotImplementedError

	def get_dict(self, RI, RW):
		feed_dict = {self.training: False}
		if RI is not None:
			feed_dict.update({self.RI: RI})
		if RW is not None:
			feed_dict.update({self.RW: RW})
		return feed_dict

	def get_P(self, sess, RI, RW):
		feed_dict = self.get_dict(RI, RW)
		P = tf.square(self.Q) / tf.reduce_sum(self.Q, axis=0)
		return sess.run(P / tf.reduce_sum(P, axis=1, keep_dims=True), feed_dict=feed_dict)

	def get_embedding(self, sess, RI, RW):
		feed_dict = self.get_dict(RI, RW)
		return sess.run(self.Z_transform, feed_dict=feed_dict)

	def predict(self, sess, RI=None, RW=None):
		feed_dict = self.get_dict(RI, RW)
		return sess.run(tf.one_hot(tf.argmax(self.Q, axis=1), depth=self.paras.num_cluster, on_value=1, off_value=0), feed_dict=feed_dict)
