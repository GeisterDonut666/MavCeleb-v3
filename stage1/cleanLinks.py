"""
M. Saad Saeed
18F-MS-CP-01
"""


import glob
import shutil
import os

count = 0
folds = []
for ids in glob.glob('identities\\*\\*\\'):
    print('Containds folders: ',ids)
    if count%2 == 0:
        folds.append(ids.split('\\')[0]+'\\'+ids.split('\\')[1])
    count+=1

if not os.path.exists('doubleFolder'):
    os.mkdir('doubleFolder')


for ids in folds:
    try:
        shutil.move(ids, 'doubleFolder\\'+ids)
    except:
        print('Already moved')


for ids in glob.glob('doubleFolder\\*\\*.txt'):
    print(ids)