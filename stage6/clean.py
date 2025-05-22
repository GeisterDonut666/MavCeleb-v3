"""
M. Saad Saeed
18F-MS-CP-01
"""
import os.path
from glob import glob
from shutil import copy
from os import path, makedirs
from subprocess import call

def data():

    acc = []           
    for ids in glob('facetracks_checked/*')[5:]:
        for typ in glob(ids+'/*'):
            for links in glob(typ+'/*/*.txt'):
                tmpdata = []
                with open(links, 'r+') as file:
                    for dat in file:    
                        tmpdata.append(dat)
                tmpacc = float(tmpdata[-1].split(':')[-1])
                if tmpacc > .25:
                    acc.append([links.split('.')[0].replace('facetracks_checked', ''), tmpacc])
    return acc


acc = data()
print(acc)

for dat in acc[0]:
    cpyDir = 'facetracks/'+dat[0]+'.avi'
    txtDir = 'facetracks/'+dat[0]+'.txt'

    try:
        newDirV = 'cleanVid/'+dat[0]
        if not path.exists('cleanVid/'+dat[0][0:-6]):
            makedirs('cleanVid/'+dat[0][0:-6])
        copy(cpyDir, newDirV+'.avi')
        copy(txtDir, newDirV+'.txt')
    except:
        print('Already Copied')


    try:
        newDirA = dat.replace('Vid', 'Wav')
        if not path.exists('cleanWav/'+dat[0][0:-6]):
            makedirs('cleanWav/'+dat[0][0:-6])
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s.wav" % (cpyDir,newDirA))
        call(command, shell=True, stdout=None)
    except:
        print('Already extracted')

    print(dat[0])
