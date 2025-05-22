import os.path
from glob import glob
from os import path, makedirs
from subprocess import call


acc = glob(os.path.join('cleanVid', '**', '*.avi'), recursive=True)
#print(acc)

for dat in acc:
    cpyDir = dat
    print(cpyDir)

    try:
        newDirA = dat.replace('Vid', 'Wav').replace('.avi', '')
        if not path.exists(newDirA):
            makedirs(newDirA)
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s.wav" % (cpyDir,newDirA))
        call(command, shell=True, stdout=None)
        os.removedirs(newDirA)

    except:
        print('Already extracted')

    print(dat[0])
