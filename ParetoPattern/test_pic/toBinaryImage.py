import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

file_dir = r'D:\workspace\DLMOP\ParetoPattern\test_pic'
file_dist_dir = r'D:\workspace\DLMOP\ParetoPattern\test_pic\test'
print('Input the filename: ')
file_name = input()

while(file_name != '0'):
    
    path = os.path.join(file_dir, file_name)
    path_dist = os.path.join(file_dist_dir, file_name[: -3] + 'dat')
    img = plt.imread(path)
    img = np.asarray(img[:, :, 0], dtype='float32')
    img /= 255.
    img = np.ones_like(img, dtype='float32') - img

    fp = open(path_dist, 'wb')
    pickle.dump(img, fp)
    fp.close()

    print(path_dist + '\n')

    print('Input the filename: ')
    file_name = input()
