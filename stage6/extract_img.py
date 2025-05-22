from glob import glob
from os import path, makedirs
from subprocess import call
import cv2
import os.path
from glob import glob
from os import path, makedirs
from subprocess import call


#print(acc)

x = 224
y = 224
save_every_n = 25

acc = glob(os.path.join('cleanVid', '**', '*.avi'), recursive=True); acc.sort()

for dat in acc[:]:
    cpyDir = dat
    #print(cpyDir)

    try:
        newDirA = dat.replace('Vid', 'Img').replace('.avi', '')
        if not path.exists(newDirA):
            makedirs(newDirA)

        cap = cv2.VideoCapture(dat)
        tframes = int(cap.get(7))

        allframes = []

        #print(tframes)
        if tframes >= 10000:
            print(newDirA)
            print(tframes)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            elif tframes % save_every_n == 0:
                #frame = frame[32:x - 32, 32:y - 32]    # Uncomment to change img output size from 224 to 160 (i think)
                cv2.imwrite(
                    newDirA + '%04d.jpg' % tframes,
                    frame)
            tframes = tframes - 1

        os.removedirs(newDirA)

    except:
        print('Already extracted')

