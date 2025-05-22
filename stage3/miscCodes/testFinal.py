"""
M. Saad Saeed
18F-MS-CP-01
"""

import cv2
import numpy as np
from sklearn.preprocessing import normalize
import glob
import os
import shutil
from multiFaceDetection import createNetwork, faceDetect
from removeNoFace import setDataSet



if 'rnet' not in globals():
    print('Creating Network')
    pnet, onet, rnet = createNetwork()
parent = 'testFinal\\'
child = 'abc\\'
try:
    shutil.rmtree(child)
except:
    print('Deleting')
os.mkdir(child)
identityList = glob.glob(parent+'*')
for identity in identityList:
    fold1 = identity.split('\\')[1]
    foldChildId = child+fold1
    os.mkdir(foldChildId)   
    idx = identity+'\\*'
    linkList = glob.glob(idx)
    for video in linkList:
        fold2  = video.split("\\")[2]
        foldChildVid = foldChildId+'\\'+fold2
        os.mkdir(foldChildVid)
        vidx =  video+'\\*'
        vidList = glob.glob(vidx)
        cap = cv2.VideoCapture(vidList[0])
        ret, current_frame = cap.read()
        faceCheck = []
        bb = []
        previous_frame = current_frame.copy()    
        count = 0
        diff = []
        while(cap.isOpened()):
            print(count)
            diff.append(np.sum(cv2.absdiff(current_frame,previous_frame)))
            flag,bounding_box,scaled = faceDetect(previous_frame,pnet,onet,rnet)
            if np.sum(scaled):
                cv2.imwrite(foldChildVid+'\\face_{:05}.jpg'.format(count),scaled)
            faceCheck.append(int(flag))
            bb.append(bounding_box)
            previous_frame = current_frame.copy()
            ret, current_frame = cap.read()
            if not ret:
                print('return')
                cap.release()
                break
            count+=1     
        diff = np.asarray(diff)
        bb = np.asarray(bb)
        faceCheck = np.asarray(faceCheck)
        diff = normalize(diff[:,np.newaxis], axis=0).ravel()
        diff = np.floor(diff*10)
        
        
        cap = cv2.VideoCapture(vidList[0])
        scount=0
        fcount=0
        with open(foldChildVid+'\\'+"{:04d}.txt".format(scount),"w+") as file:
            file.write("Identity: %s \n" % (fold1))
            file.write("Reference: %s \n\n" % (foldChildVid))
            file.write("Frames\tX\tY\tW\tH\n")
        with open (foldChildVid+'\\SceneInfo.txt','+w') as file:
            file.write("Identity: %s \n" % (fold1))
            file.write("Reference: %s \n\n" % (foldChildVid))
            file.write('S\tF1\tF2\n')
            file.write('{:03d}\t{:04d}\t'.format(scount,fcount))
        while (cap.isOpened()):
            ret, currentFrame = cap.read()
            if fcount==0:
                cv2.imwrite(foldChildVid+"\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
            bbTemp = bb[fcount]
            with open(foldChildVid+'\\'+"{:04d}.txt".format(scount),"a+") as file:
                file.write("{:05d}\t{:d}\t{:d}\t{:d}\t{:d}\n".format(fcount,bbTemp[1],
                                                     bbTemp[0],
                                                     bbTemp[3]-
                                                     bbTemp[1],
                                                     bbTemp[2]-
                                                     bbTemp[0]))
            if diff[fcount]>=1:
                cap.set(1,fcount-1)
                ret, currentFrame = cap.read()
                cv2.imwrite(foldChildVid+"\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount-1),currentFrame)
                with open(foldChildVid+'\\'+"{:04d}.txt".format(scount),"r") as file:
                    temp = file.readlines()
#                    temp = temp[:-1]
                with open(foldChildVid+'\\'+"{:04d}.txt".format(scount),"+w") as file:
                    for dat in temp:    
                        file.write(dat)
                scount+=1
                with open (foldChildVid+'\\SceneInfo.txt','+a') as file:
                    file.write('{:04d}\t\n'.format(fcount-1))
                    file.write('{:03d}\t{:04d}\t'.format(scount,fcount))
                bbTemp = bb[fcount+1]
                with open(foldChildVid+'\\'+"{:04d}.txt".format(scount),"w+") as file:
                    file.write("Identity: %s \n" % (fold1))
                    file.write("Reference: %s \n\n" % (foldChildVid))
                    file.write("Frames\tX\tY\tW\tH\n")
                    file.write("{:05d}\t{:d}\t{:d}\t{:d}\t{:d}\n".format(fcount+1,bbTemp[1],
                                                 bbTemp[0],
                                                 bbTemp[3]-
                                                 bbTemp[1],
                                                 bbTemp[2]-
                                                 bbTemp[0]))
                ret,currentFrame = cap.read()
                cv2.imwrite(foldChildVid+"\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
                print(fcount)
                fcount+=1
            fcount+=1
            if fcount==len(diff):
                cv2.imwrite(foldChildVid+"\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
                with open (foldChildVid+'\\SceneInfo.txt','+a') as file:
                    file.write('{:04d}\n'.format(fcount-1))
                cap.release()
                break
 
setDataSet()           



# TODO[1]:
'''
Error in storing txt file: one ahead
Error in storing edges: one previous frame no
'''