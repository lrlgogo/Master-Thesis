from PFRecAlg import *
import time
import os
import pickle


file_dir = r'D:\workspace\DLMOP\ParetoPattern\test_pic\test'
file_name = 'test_2.dat'


def fun(u=13, file_name=file_name, file_dir=file_dir):

    fp = open(os.path.join(file_dir, file_name), 'rb')

    A = pickle.load(fp)
    fp.close()

    f = PFRecAlgII(u)
    f.get_input(A)
    time_sta = time.time()
    f.run()
    time_end = time.time()
    print('time used: ',time_end - time_sta)
    # f.show()

    y = f.output
    plt.subplot(121)
    plt.imshow(A, cmap='Greys')
    plt.subplot(122)
    plt.imshow(y, cmap='Greys')
    plt.show()

    return y, time_end, time_sta
