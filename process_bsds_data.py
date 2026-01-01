import numpy as np
# import matplotlib.pyplot as plt
from scipy import io
import scipy
import os
from PIL import Image
PATH = "/home/tjq/下载/BSDS500/gt/"

"""处理test"""
test_list = os.listdir(PATH + 'test')
print(len(test_list))
for index in test_list:
    name = index.split('.')[0]
    print(name)
    test = io.loadmat(PATH + '/test/' + index)
    # print(train)
    a = np.array(1024)
    a = test['groundTruth'][0][0][0][0][1]
    print(a)
    a = a * 255
    print(PATH + 'trans/test/' + str(name))
    Image.fromarray(a).save(PATH + 'trans/test/' + str(name) + '.jpg')
