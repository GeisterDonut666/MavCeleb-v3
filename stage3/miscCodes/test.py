"""
M. Saad Saeed
18F-MS-CP-01
"""
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import os
import shutil
from multiFaceDetection import createNetwork, faceDetect

if 'rnet' not in globals():
    print('Creating Network')
    pnet, onet, rnet = createNetwork()
vid = 'vid.mp4'
cap = cv2.VideoCapture(vid)
ret, current_frame = cap.read()
previous_frame = current_frame

count = 0
diff = []

while(cap.isOpened()):
    print(count)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
#    current_hist = cv2.calcHist([current_frame], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#    current_hist = cv2.normalize(current_hist, current_hist).flatten()
#    previous_hist = cv2.calcHist([previous_frame], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#    previous_hist = cv2.normalize(previous_hist, previous_hist).flatten()
#    diff.append(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL))
#    diff.append(np.sum(np.average(np.bitwise_xor(current_frame_gray,previous_frame_gray))))
#    diff.append(np.bitwise_xor(current_frame_gray,previous_frame_gray))
    diff.append(np.sum(cv2.absdiff(current_frame,previous_frame)))
    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()
    if not ret:
        print('return')
        cap.release()
        break
    count+=1

    
diff = np.asarray(diff)
diff = normalize(diff[:,np.newaxis], axis=0).ravel()
diff = np.floor(diff*10)
cap = cv2.VideoCapture(vid)
ret,initial = cap.read()

scount=1
fcount=0
sceneDir = 'boundaries\\Scene'+str(scount)
try:
    os.mkdir(sceneDir)
except:
    shutil.rmtree(sceneDir)
    os.mkdir(sceneDir)

#cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount+1),initial)
with open(sceneDir+".txt","a+") as file:
            file.write("Reference: %s \n\n" % (sceneDir))
            file.write("Frames\tX\tY\tW\tH\tAcc\n")
with open ('boundaries\\scene.txt','+w') as file:
    file.write('S\tF1\tF2\n')
    file.write('{:03d}\t{:04d}\t'.format(scount,fcount+1))
    
check,bb = faceDetect(initial,sceneDir,fcount,pnet,onet,rnet)

while (cap.isOpened()):
    ret, currentFrame = cap.read()
    if diff[fcount]>=1:
        cap.set(1,fcount-1)
        ret, currentFrame = cap.read()
        
#        cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
        
        with open ('boundaries\\bound.txt','+a') as file:
            file.write('{:04d}\t\n'.format(fcount))
        scount+=1
        sceneDir = 'boundaries\\Scene'+str(scount)
        try:
            os.mkdir(sceneDir)
        except:
            shutil.rmtree(sceneDir)
            os.mkdir(sceneDir)
        ret, currentFrame = cap.read()
        
#        cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount+1),currentFrame)
        
        with open ('boundaries\\bound.txt','+a') as file:
            file.write('{:03d}\t{:04d}\t'.format(scount,fcount+1))
        print(fcount)
    fcount+=1
    if fcount>len(diff)-1:
        
#        cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
        
        with open ('boundaries\\bound.txt','+a') as file:
            file.write('{:04d}\n'.format(fcount))
        cap.release()
        break