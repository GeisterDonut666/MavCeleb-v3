# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:16:56 2019

@author: DrHaroonYousaf
"""

from glob import glob
from os import listdir, walk, path
from shutil import move
from random import shuffle


#if not path.exists('D:\\Saad\\fiveStepsAlgo\\stage2\\downVids\\poi.txt'):
iden = []
for ids in glob('D:\\Saad\\fiveStepsAlgo\\stage2\\downVids\\U_E\\both\\*'):
    typ = next(walk(ids))[1]
    lenE = len(listdir(ids+'\\'+typ[0]))
    lenU = len(listdir(ids+'\\'+typ[1]))
    if (lenU>=3 and lenE>=3)==True:
        iden.append(ids)
#    for i in range(20):
#        shuffle(iden)
#    iden1 = iden[0:25]
#    iden2 = iden[-26:-1]
#    iden = []
#    for ids in iden1:
#        iden.append(ids)
#    for ids in iden2:
#        iden.append(ids)
#    with open('poi.txt','a+') as file:
#        for ids in iden:
#            file.write(ids.split('\\')[2]+'\n')
#else:
#    print('POI extracted')


#
#if len(listdir('data50'))<50:
#    iden = []
#    with open('D:\\Saad\\fiveStepsAlgo\\stage2\\downVids\\poi.txt') as file:
#        for ids in file:
#            iden.append('D:\\Saad\\fiveStepsAlgo\\stage2\\downVids\\U_E\\both\\'+ids.split('\n')[0])
#            src = 'D:\\Saad\\fiveStepsAlgo\\stage2\\downVids\\U_E\\both\\'+ids.split('\n')[0]
#            dst = 'data50\\'+ids.split('\n')[0]
#            move(src,dst)
#else:
#    print('Already Copied')      