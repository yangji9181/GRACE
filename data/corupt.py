import random

paths = ['cora/', 'citeseer/', 'facebook/0/', 'facebook/107/', 'facebook/348/', 'facebook/414/', 'facebook/686/', 'facebook/698/', 'facebook/1684/', 'facebook/1912/', 'facebook/3437/', 'facebook/3980/']
rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for p in paths:
	fin = open(p+'feature.txt', 'r')
	feats = []
	for line in fin:
		feats.append([int(i) for i in line.strip().split(',')])
	fin.close()

	for r_f in rates:
		r = r_f * 1.5
		fout = open(p+'feature/'+'flip_'+str(r_f)+'.txt', 'w')
		for f in feats:
			for i in range(len(f) - 1):
				if random.random() < r:
					fout.write(str(1 - f[i])+',')
				else:
					fout.write(str(f[i])+',')
			if random.random() < r:
				fout.write(str(1 - f[-1])+'\n')
			else:
				fout.write(str(f[-1])+'\n')
		fout.close()

		fout = open(p+'feature/'+'remove_'+str(r)+'.txt', 'w')
		for f in feats:
			for i in range(len(f) - 1):
				if f[i] == 1:
					if random.random() < r:
						fout.write('0,')
					else:
						fout.write('1,')
				else:
					fout.write('0,')
			if f[-1] == 1:	
				if random.random() < r:
					fout.write('0\n')
				else:
					fout.write('1\n')
			else:
				fout.write('0\n')
		fout.close()




