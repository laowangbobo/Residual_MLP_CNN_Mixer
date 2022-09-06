# by Yang Haibo(2022/1/11)
# yanghaibo4869@gmail.com

import os

'''
This code is used to remove some unrelated files.
It can avoid some file address conflict error.
'''

# unrelated files address
path1 = '../data/dataset/Free/test.txt'
path2 = 'loss/test.txt'
path3 = 'model/test.txt'
path4 = 'result/test.txt'

# delete unrelated files
if os.path.exists(path1):
    os.remove(path1) 

if os.path.exists(path2):
    os.remove(path2) 

if os.path.exists(path3):
    os.remove(path3) 

if os.path.exists(path4):
    os.remove(path4) 