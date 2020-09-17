# thread limit
import os
os.environ["OMP_NUM_THREADS"] = "8"

import sys
import numpy as np
import gc
from multiprocessing import Pool
from multiprocessing import Process, Queue
import time
from tqdm import tqdm
import datetime

def rho(x, u, N, sigma):
	rslt = np.array([])
	Nx = len(x)
	for num in range(Nx):
		vec = x[num,np.newaxis]-u[np.newaxis,:]
		s1 = np.exp(np.linalg.norm(vec, axis=2)**2/(-2)/sigma)
		s2 = np.sum(s1, axis=1)
		rslt = np.append(rslt, s2/N/np.sqrt((2*np.pi*sigma)**3))
	return rslt

def wrapper_rho(args):
	return rho(*args)

def pfk(Dg1, Dg2, tm, dtau):
	"""
	Dg1: diagram 1
	Dg2: diagram 2
	tm: max time

	return
	kPF
	"""
	
	#noisesh = 0 # ノイズどこで切るか（これもともと0でやってなかった？0でやってたから0で）
	#Dg1 = np.array([i for i in Dg1 if int(i[2]*10)%int(dtau*10)==0 and i[2]<tm+0.001 and i[1]-i[0]>=noisesh])
	#Dg2 = np.array([i for i in Dg2 if int(i[2]*10)%int(dtau*10)==0 and i[2]<tm+0.001 and i[1]-i[0]>=noisesh])
	Dg1 = np.array([i for i in Dg1 if i[2]<tm+0.001])
	Dg2 = np.array([i for i in Dg2 if i[2]<tm+0.001])
	
	# projection
	sigma = 0.001
	xi = 0.1
	N1 = len(Dg1)
	N2 = len(Dg2)
	N = N1+N2

	Dg1[:,2] = xi*Dg1[:,2]
	Dg2[:,2] = xi*Dg2[:,2]
	
	Dgd1 = np.zeros((N1,3))
	Dgd2 = np.zeros((N2,3))

	Dgd1[:,0] = (Dg1[:,0]+Dg1[:,1])/2
	Dgd1[:,1] = (Dg1[:,0]+Dg1[:,1])/2
	Dgd1[:,2] = Dg1[:,2]
	Dgd2[:,0] = (Dg2[:,0]+Dg2[:,1])/2
	Dgd2[:,1] = (Dg2[:,0]+Dg2[:,1])/2
	Dgd2[:,2] = Dg2[:,2]
	
	p = np.vstack((Dg1, Dgd2))
	q = np.vstack((Dg2, Dgd1))
	
	del Dg1, Dg2, xi, N1, N2, Dgd1, Dgd2
	#del Dg1, Dg2, N1, N2, Dgd1, Dgd2
	gc.collect()
	
	# Fisher imformation geometry	
	pl = Pool(processes=4)
	
	S = 4 # split
	
	var_list_pp = [(p[i*N//S:(i+1)*N//S],p,N,sigma) for i in range(S)]
	var_list_qp = [(q[i*N//S:(i+1)*N//S],p,N,sigma) for i in range(S)]
	var_list_pq = [(p[i*N//S:(i+1)*N//S],q,N,sigma) for i in range(S)]
	var_list_qq = [(q[i*N//S:(i+1)*N//S],q,N,sigma) for i in range(S)]
	
	convec = np.concatenate([var_list_pp, var_list_qp, var_list_pq, var_list_qq])
	
	del sigma, var_list_pp, var_list_qp, var_list_pq, var_list_qq, p, q, N
	gc.collect()
	
	figpq = pl.map(wrapper_rho, convec)
	
	x = np.concatenate([figpq[i] for i in range(2*S)])
	y = np.concatenate([figpq[2*S + i] for i in range(2*S)])

	pl.close()
	del S, figpq
	gc.collect()

	x = x/np.linalg.norm(x)
	y = y/np.linalg.norm(y)

	t = 1 # 感度
	d = np.dot(x,y) if np.dot(x,y) < 1.0 else 1.0

	kPF = np.exp(-1*t*np.arccos(np.sqrt(d)))

	del x, y, t, d
	gc.collect()

	return kPF

def gm_normalize(gm,sp):
	gmn = np.ones((sp,sp))
	for i in range(sp-1):
		for j in range(i+1,sp):
			rt = np.sqrt(gm[i][i])*np.sqrt(gm[j][j])
			gmn[i][j] = gm[i][j]/rt
			gmn[j][i] = gm[i][j]/rt
			
	return gmn
	
if __name__ == '__main__':
	sp = 300 # サンプル数
	div_lst = [1,2,4]
	divsp = sp//len(div_lst) # div１こあたりのサンプル数
	dim = [0,1] # n-th homology
	
	#load data
	print('loading... {0}'.format(datetime.datetime.now()))
	PH_data = [np.loadtxt('freqgauss/torus_test/div1/freqgauss_torustest_div1_dim{0}_{1}.txt'.format(dim[0],i)) for i in range(divsp)]\
			+ [np.loadtxt('freqgauss/torus_test/div2/freqgauss_torustest_div2_dim{0}_{1}.txt'.format(dim[0],i)) for i in range(divsp)]\
			+ [np.loadtxt('freqgauss/torus_test/div4/freqgauss_torustest_div4_dim{0}_{1}.txt'.format(dim[0],i)) for i in range(divsp)]
	print('data loaded {0}'.format(datetime.datetime.now()))
	
	### calculate kernel values
	for dtau in [0.8]:#時間幅
		noisesh = 0.01
		PH_data = np.array([[i for i in item if int(i[2]*10)%int(dtau*10)==0 and i[1]-i[0]>=noisesh] for item in PH_data])
		for stept in range(125): # 最大時間
			max_time = stept*dtau
			print('step:{0} max_time:{1} start'.format(stept,max_time), datetime.datetime.now())
			gm = np.ones((sp,sp))

			for i in tqdm(range(sp-1)):
				for j in range(i+1,sp):
					gm[i][j] = pfk(PH_data[i], PH_data[j], max_time, dtau)

			### normalization
			gm = gm_normalize(gm,sp)

			np.savetxt('freqgauss/torus_test/gm/fast_freqgauss_torustest_gmcut_dim{0}_{1}.txt'.format(dim[0],stept), gm)
						
		print('end', datetime.datetime.now())
