import argparse
import getpass
import sys


class Config():
	def __init__(self, args):
		'''
		convert Namespace to Config object
		:param args:
		'''
		var = vars(args)
		for k, v in var.items():
			setattr(self, k, v)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--devices', type=list, default=[2, 3], help='Available GPU')
	parser.add_argument('--num_exp', type=int, default=4, help='Number of experiment')
	parser.add_argument('--num_device', type=int, default=2, help='Number of GPU, change to 0 if not using CPU')
	parser.add_argument('--device', type=int, default=-1, help='Device id')
	parser.add_argument('--gpu_memory_fraction', type=float, default=1.0, help='fraction of gpu memory per process')
	parser.add_argument('--batch_gpu_process', type=int, default= 1, help='Number of processes allowed on one GPU')
	parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
	parser.add_argument('--feat_dim', type=int, default=-1, help='Feature dimension')
	parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
	parser.add_argument('--encoder_hidden', type=list, default=[256], help='Encoder hidden layer dimension')
	parser.add_argument('--decoder_hidden', type=list, default=[256], help='Decoder hidden layer dimension')
	parser.add_argument('--transition_function', type=str, default='RI', help='Transition function [T, RI, RW]')
	parser.add_argument('--random_walk_step', type=int, default=2, help=None)
	parser.add_argument('--alpha', type=float, default=0.9, help='Damping coefficient for propagation process')
	parser.add_argument('--lambda_', type=float, default=0.1)
	parser.add_argument('--keep_prob', type=float, default=0.4, help='Keep probability of dropout')
	parser.add_argument('--BN', type=bool, default=False, help='Apply batch normalization')
	parser.add_argument('--lambda_r', type=float, default=1.0, help='Reconstruct loss coefficient')
	parser.add_argument('--lambda_c', type=float, default=0.2, help='Clustering loss coefficient')
	parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer [Adam, Momentum, GradientDescent, RMSProp, Adagrad]')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--pre_epoch', type=int, default=1, help=None)
	parser.add_argument('--pre_step', type=int, default=1, help=None)
	parser.add_argument('--epoch', type=int, default=1, help=None)
	parser.add_argument('--step', type=int, default=1, help=None)
	parser.add_argument('--epsilon', type=float, default=1.0, help='Annealing hyperparameter for cluster assignment')
	parser.add_argument('--dataset', type=str, default='pubmed', help=None)
	parser.add_argument('--dense_graph', type=bool, default=True, help='Set to True when using large graph')
	return parser.parse_args()


def init_dir(args):
	args.data_dir = base_dir(args)
	args.model_dir = args.data_dir + 'model/'
	args.feature_file = args.data_dir + 'feature.txt'
	args.edge_file = args.data_dir + 'edge.txt'
	args.cluster_file = args.data_dir + 'cluster.txt'
	args.model_file = args.data_dir + 'model.pkl'
	args.plot_file = args.data_dir + 'plot.png'
	args.predict_file = args.data_dir + 'prediction.txt'

def base_dir(args):
	return 'data/' + args.dataset + '/' if sys.platform == 'darwin' else \
		'/shared/data/' + getpass.getuser() + '/DEC/' + args.dataset + '/'


args = parse_args()
init_dir(args)
