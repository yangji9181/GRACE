from __future__ import print_function

import os
import subprocess
from copy import deepcopy
from multiprocessing import *

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from GRACE import *
from config import *
from evaluate import f1_community, jc_community, nmi_community
from utils import *


class Predictor(object):
	def __init__(self, paras):
		self.paras = deepcopy(paras)
		self.graph = load_graph(paras.data_dir, paras.feature_file, paras.edge_file, paras.cluster_file, paras.alpha, paras.lambda_)
		self.reset_paras()

	def reset_paras(self):
		self.paras.feat_dim = len(self.graph.feature[0])
		self.paras.num_node = len(self.graph.feature)
		self.paras.num_cluster = len(self.graph.cluster[0])

	def batch(self):
		batch_indices = np.random.randint(self.paras.num_node, size=self.paras.batch_size)
		RI = self.graph.RI[:, batch_indices] if self.paras.transition_function == 'RI' else None
		RW = self.graph.RW[:, batch_indices] if self.paras.transition_function == 'RW' else None
		return batch_indices, RI, RW

	def fit(self, model, sess):
		for _ in tqdm(range(self.paras.pre_epoch), ncols=100):
			for _ in range(self.paras.pre_step):
				sess.run(model.pre_gradient_descent, feed_dict={model.training: True})
		print('reconstruction loss: %f' % sess.run(model.loss_r, feed_dict={model.training: False}))

		Z = sess.run(model.Z_transform, feed_dict={model.training: False})
		kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(Z)
		model.init_mean(kmeans.cluster_centers_, sess)

		self.diff = []
		s_prev = model.predict(sess)
		for _ in tqdm(range(self.paras.epoch), ncols=100):
			P = model.get_P(sess)
			for _ in range(self.paras.step):
				sess.run(model.gradient_descent, feed_dict={model.training: True, model.P: P})
			s = model.predict(sess)
			self.diff.append(np.sum(s_prev != s) / 2.0)
			s_prev = s
		P = model.get_P(sess)
		print('reconstruction loss: %f' % sess.run(model.loss_r, feed_dict={model.training: False}))
		print('clustering loss: %f' % sess.run(model.loss_c, feed_dict={model.training: False, model.P: P}))
		self.embedding = model.get_embedding(sess)
		self.prediction = model.predict(sess)

	def feed_dict(self, model, RI, RW):
		feed_dict = {}
		if RI is not None:
			feed_dict.update({model.RI: RI})
		if RW is not None:
			feed_dict.update({model.RW: RW})
		return feed_dict

	def fit_dense(self, model, sess):
		for _ in tqdm(range(self.paras.pre_epoch), ncols=100):
			for _ in range(self.paras.pre_step):
				_, RI, RW = self.batch()
				feed_dict = {model.training: True}
				feed_dict.update(self.feed_dict(model, RI, RW))
				sess.run(model.pre_gradient_descent, feed_dict=feed_dict)
		RI, RW = self.graph.RI, self.graph.RW
		feed_dict = {model.training: False}
		feed_dict.update(self.feed_dict(model, RI, RW))
		print('reconstruction loss: %f' % sess.run(model.loss_r, feed_dict=feed_dict))

		Z = sess.run(model.Z_transform, feed_dict=feed_dict)
		kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(Z)
		model.init_mean(kmeans.cluster_centers_, sess)

		self.diff = []
		s_prev = model.predict(sess, RI, RW)
		for _ in tqdm(range(self.paras.epoch), ncols=100):
			RI, RW = self.graph.RI, self.graph.RW
			P = model.get_P(sess, RI, RW)
			for _ in range(self.paras.step):
				batch_indices, RI, RW = self.batch()
				feed_dict = {model.training: True, model.P: P[batch_indices]}
				feed_dict.update(self.feed_dict(model, RI, RW))
				sess.run(model.gradient_descent, feed_dict=feed_dict)
			RI, RW = self.graph.RI, self.graph.RW
			s = model.predict(sess, RI, RW)
			self.diff.append(np.sum(s_prev != s) / 2.0)
			s_prev = s

		RI, RW = self.graph.RI, self.graph.RW
		P = model.get_P(sess, RI, RW)
		feed_dict = {model.training: False, model.P: P}
		feed_dict.update(self.feed_dict(model, RI, RW))
		print('reconstruction loss: %f' % sess.run(model.loss_r, feed_dict=feed_dict))
		print('clustering loss: %f' % sess.run(model.loss_c, feed_dict=feed_dict))
		self.embedding = model.get_embedding(sess, RI, RW)
		self.prediction = model.predict(sess, RI, RW)

	def train(self):
		tf.reset_default_graph()
		if self.paras.device >= 0:
			os.environ['CUDA_VISIBLE_DEVICES'] = str(self.paras.device)
			with tf.device('/gpu:0'):
				model = GRACE_Dense(self.paras, self.graph) if self.paras.dense_graph else GRACE(self.paras, self.graph)
		else:
			os.environ['CUDA_VISIBLE_DEVICES'] = ''
			with tf.device('/cpu:0'):
				model = GRACE_Dense(self.paras, self.graph) if self.paras.dense_graph else GRACE(self.paras, self.graph)
		with tf.Session(config=tf.ConfigProto(
				allow_soft_placement=True,
				gpu_options=tf.GPUOptions(
					per_process_gpu_memory_fraction=self.paras.gpu_memory_fraction,
					allow_growth=True))) as sess:
			tf.summary.FileWriter(self.paras.model_dir, graph=sess.graph)
			sess.run(tf.global_variables_initializer())
			if self.paras.dense_graph:
				self.fit_dense(model, sess)
			else:
				self.fit(model, sess)
		os.environ['CUDA_VISIBLE_DEVICES'] = ''

	def plot(self):
		scatter(self.tSNE(), np.argmax(self.graph.cluster, axis=1), self.paras.plot_file)
		# plot(self.diff, self.paras.plot_file)

	def evaluate(self):
		prediction, ground_truth = np.transpose(self.prediction), np.transpose(self.graph.cluster)
		return f1_community(prediction, ground_truth), jc_community(prediction, ground_truth), nmi_community(prediction, ground_truth)

	def dump(self):
		with open(self.paras.predict_file, 'w') as f:
			for prediction in self.prediction:
				f.write(','.join(map(str, prediction)) + '\n')

	def tSNE(self):
		return TSNE(n_components=2).fit_transform(self.embedding)


def initialize_predictors(args):
	predictors = []
	if args.dataset in ['facebook', 'twitter', 'gplus']:
		data_dir = base_dir(args)
		dataset = args.dataset
		processes = []
		queue = Queue()
		for subdir in os.listdir(data_dir):
			if subdir == '.DS_Store':
				continue
			args.dataset = dataset + '/' + subdir
			init_dir(args)
			subprocess.call('rm ' + args.model_dir + '*', shell=True)
			process = Process(target=lambda : queue.put(Predictor(args)))
			process.start()
			processes.append(process)
		for _ in processes:
			predictors.append(queue.get())
		for process in processes:
			process.join()
	else:
		subprocess.call('rm ' + args.model_dir + '*', shell=True)
		predictors.append(Predictor(args))
	return predictors
