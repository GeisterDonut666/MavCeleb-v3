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


parent = 'dataSet1\\'
child = 'tracked\\'

identityList = glob.glob(parent+'*')
for identity in identityList:
    fold1 = identity.split('\\')[1]
    foldChildId = child+fold1
    try:
        os.mkdir(foldChildId)
    except OSError:
        shutil.rmtree(foldChildId)
        try:
            os.mkdir(foldChildId)
        except OSError:
            print('Error')    
    idx = identity+'\\*'
    linkList = glob.glob(idx)
    for video in linkList:
        fold2  = video.split("\\")[2]
        foldChildVid = foldChildId+'\\'+fold2
        try:
            os.mkdir(foldChildVid)
        except OSError:
            shutil.rmtree(foldChildVid)
            try:
                os.mkdir(foldChildVid)
            except OSError:
                print('Error')
                
        detFaceDirect = foldChildVid+'\\'+'detectedFaces'
        try:
            os.mkdir(detFaceDirect)
        except:
            print('Error')   
#        with open(fold1+".txt","a+") as file:
#            file.write("Reference: %s \n\n" % (fold2))
#            file.write("Frames\tX\tY\tW\tH\tAcc\n")
        '''
        Detect scene boundaries by comparing consecutive frames using CHD
        '''
        vidx =  video+'\\*'
        vidList = glob.glob(vidx)
        cap = cv2.VideoCapture(vidList[0])
        i =0
        ret, current_frame = cap.read()
        previous_frame = current_frame
        count = 0
        diff = []
        while(cap.isOpened()):
            print(count)    
            current_hist = cv2.calcHist([current_frame], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
            current_hist = cv2.normalize(current_hist, current_hist).flatten()
            previous_hist = cv2.calcHist([previous_frame], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
            previous_hist = cv2.normalize(previous_hist, previous_hist).flatten()
            diff.append(cv2.compareHist(current_hist, previous_hist, cv2.HISTCMP_BHATTACHARYYA ))
            previous_frame = current_frame.copy()
            ret, current_frame = cap.read()
            if not ret:
                print('return')
                cap.release()
                break
            count+=1
        diff = np.asarray(diff)
        diff = normalize(diff[:,np.newaxis], axis=0).ravel()
        diff = np.round(diff*10)
        cap = cv2.VideoCapture(vidList[0])
        ret,initial = cap.read()
        '''
        Now store each frame in designated id, link and scene folder
        '''
        scount=1
        fcount=0
        #name = ;
#        cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount+1),initial)
        with open (detFaceDirect+'\\bound.txt','+w') as file:
            file.write('S\tF1\tF2\n')
            file.write('{:03d}\t{:04d}\t'.format(scount,fcount+1))
        while (cap.isOpened()):
            ret, currentFrame = cap.read()
            if diff[fcount]>=1:
                
                sceneDir = detFaceDirect+'\\'+'Scene'+str(scount)
                try:
                    os.mkdir(sceneDir)
                except:
                    print('')
                    
                cap.set(1,fcount-1)
                ret, currentFrame = cap.read()
#                cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
                with open (detFaceDirect+'\\bound.txt','+a') as file:
                    file.write('{:04d}\t\n'.format(fcount))
                scount+=1
                ret, currentFrame = cap.read()
#                cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount+1),currentFrame)
                with open (detFaceDirect+'\\bound.txt','+a') as file:
                    file.write('{:03d}\t{:04d}\t'.format(scount,fcount+1))
                print(fcount)
            fcount+=1
            if fcount>len(diff)-1:
#                cv2.imwrite("boundaries\\s_{:03d}_f_{:03d}.jpg".format(scount,fcount),currentFrame)
                with open (detFaceDirect+'\\bound.txt','+a') as file:
                    file.write('{:04d}\n'.format(fcount+1))
                cap.release()
                break