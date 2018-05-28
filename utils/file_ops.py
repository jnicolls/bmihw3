'''
	file_ops.py

'''

import h5py
import numpy as np


def read_the_data(path):
	'''
		this is some garbage but i don't want to forget the syntax 
	'''
	f = h5py.File(path + '/' + path[1] +'/' + first[1] + '/' +  files[1], 'r')
	d = np.array(f.get('data'))


fig, ax = plt.subplots(2,1)
ax[0].imshow(d[:,:,0])
ax[1].imshow(d[:,:,1])