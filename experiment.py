from __future__ import print_function

from predictor import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--embed_dim', type=list, default=[500], help='Embedding dimension')
	parser.add_argument('--encoder_hidden', type=list, default=[[], [500], [500, 500], [500, 500, 500]], help='Encoder hidden layer dimension')
	parser.add_argument('--transition_function', type=list, default=['RI'], help='Transition function [T, RI, RW]')
	parser.add_argument('--random_walk_step', type=list, default=[0], help='Number of random walk steps')
	parser.add_argument('--keep_prob', type=list, default= [0.5], help='Keep probability of dropout')
	parser.add_argument('--BN', type=list, default=[False], help='Apply batch normalization')
	parser.add_argument('--lambda_c', type=list, default=[0.2], help='Clustering loss coefficient')
	parser.add_argument('--optimizer', type=list, default=['Adam'], help='Optimizer [Adam, Momentum, GradientDescent, RMSProp, Adagrad]')
	parser.add_argument('--pre_epoch', type=list, default=[100], help=None)
	parser.add_argument('--pre_step', type=list, default=[10], help=None)
	parser.add_argument('--epoch', type=list, default=[30], help=None)
	parser.add_argument('--step', type=list, default=[30], help=None)
	return parser.parse_args()


def worker(predictors, queue, batch=args.batch_gpu_process):
	'''
	:param predictors: one predictor per ego network
	:return:
	'''
	def sub_worker(predictor):
		predictor.train()
		f1, jc, nmi = predictor.evaluate()
		sub_queue.put((f1, jc, nmi))

	f1_list, jc_list, nmi_list = [], [], []
	processes = []
	sub_queue = Queue()
	for i, predictor in enumerate(predictors):
		process = Process(target=sub_worker, args=(predictor,))
		process.start()
		processes.append(process)
		if len(processes) == batch:
			for _ in processes:
				f1, jc, nmi = sub_queue.get()
				f1_list.append(f1)
				jc_list.append(jc)
				nmi_list.append(nmi)
			for process in processes:
				process.join()
			processes = []
	for _ in processes:
		f1, jc, nmi = sub_queue.get()
		f1_list.append(f1)
		jc_list.append(jc)
		nmi_list.append(nmi)
	for process in processes:
		process.join()
	queue.put((np.mean(f1_list), np.mean(jc_list), np.mean(nmi_list)))

def run(num_exp):
	predictors = initialize_predictors(args)
	f1_list, jc_list, nmi_list = [], [], []
	queue = Queue()
	processes = []
	batch_processes = []
	for i in range(num_exp):
		device_id = -1 if args.num_device == 0 else args.devices[i % args.num_device]
		for predictor in predictors:
			predictor.paras.device = device_id
		process = Process(target=worker, args=(predictors, queue,))
		process.start()
		processes.append(process)
		if args.num_device != 0:
			batch_processes.append(process)
		if args.num_device != 0 and len(batch_processes) == args.num_device:
			for process in batch_processes:
				process.join()
			batch_processes = []
	for process in processes:
		process.join()

	for _ in processes:
		f1, jc, nmi = queue.get()
		f1_list.append(f1)
		jc_list.append(jc)
		nmi_list.append(nmi)

	return np.mean(f1_list), np.std(f1_list), np.mean(jc_list), np.std(jc_list), np.mean(nmi_list), np.std(nmi_list)

if __name__ == '__main__':
	local_args = parse_args()
	f = open('results.txt', 'w')
	for embed_dim in local_args.embed_dim:
		args.embed_dim = embed_dim
		for encoder_hidden in local_args.encoder_hidden:
			args.encoder_hidden, args.decoder_hidden = encoder_hidden, list(reversed(encoder_hidden))
			for keep_prob in local_args.keep_prob:
				args.keep_prob = keep_prob
				for BN in local_args.BN:
					args.BN = BN
					for lambda_c in local_args.lambda_c:
						args.lambda_c = lambda_c
						for optimizer in local_args.optimizer:
							args.optimizer = optimizer
							for pre_epoch in local_args.pre_epoch:
								args.pre_epoch = pre_epoch
								for pre_step in local_args.pre_step:
									args.pre_step = pre_step
									for epoch in local_args.epoch:
										args.epoch = epoch
										for step in local_args.step:
											args.step = step
											for transition_function in local_args.transition_function:
												args.transition_function = transition_function
												for random_walk_step in local_args.random_walk_step:
													args.random_walk_step = random_walk_step

													#f.write(args)
													f1_mean, f1_std, jc_mean, jc_std, nmi_mean, nmi_std = run(args.num_exp)
													f.write('f1 mean %f, std %f\n' % (f1_mean, f1_std))
													#f.write('jc mean %f, std %f\n' % (jc_mean, jc_std))
													#f.write('nmi mean %f, std %f\n' % (nmi_mean, nmi_std))
	f.close()
