# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:11:38 2019

@author: DrHaroonYousaf
"""

import glob
import numpy as np
import os
import shutil

def setDataSet():

    dest = 'boundaries\\'
    identityList = glob.glob(dest+'*')
    for identity in identityList:
        idx = identity+'\\*'
        linkList = glob.glob(idx)
        for video in linkList:
            vidx =  video+'\\0*.txt'
            scene = video+'\\s_*.jpg'
            face = video+'\\f*.jpg'
            sceneList = glob.glob(scene)
            vidList = glob.glob(vidx)
            faceList = glob.glob(face)
            noface = []
            for txt in vidList:
                pos = 0
                neg = 0
                with open(txt,'r') as file:
                    tempRead = file.readlines()
                    cords = tempRead[4:len(tempRead)]
                    temp = 0
                    for inp in range(len(cords)):
                         temp = cords[inp]
                         temp = temp.split('\t')[1:5]
                         tempSum = list(map(int, temp))
                         tempSum = np.asarray(tempSum)
                         tempSum = np.sum(tempSum)
                         if tempSum > 0:
                             pos+=1
                         else:
                             neg+=1              
                if neg > pos:
                    noface.append(txt)
                    
            newDir = video+'\\noFace\\'
            sceneCut = video+'\\SceneCuts\\'
            faceCrop = video+'\\FaceCropped\\'
            os.mkdir(newDir)
            os.mkdir(sceneCut)
            os.mkdir(faceCrop)
            for nf in noface:
                shutil.move(nf,newDir)  
            for scene in sceneList:
                shutil.move(scene,sceneCut)     
            for face in faceList:
                shutil.move(face,faceCrop)  
            shutil.move(video+'\\SceneInfo.txt',sceneCut)
            
    for identity in identityList:
        idx = identity+'\\*'
        linkList = glob.glob(idx)
        framen = []
        for video in linkList:
            faceCrop = video+'\\FaceCropped\\'
            txtTemp = glob.glob(video+'\\no*')
            txtTemp = glob.glob(txtTemp[0]+'\\*.txt')
            for txtFile in txtTemp:
                mylines = []
                with open (txtFile, 'r+') as myfile:
                    for myline in myfile: 
                        mylines.append(myline) 
                cords = mylines[4:len(mylines)]
                for inp in range(len(cords)):
                    temp = cords[inp]
                    temp = temp.split('\t')[0:5]
                    framen.append(temp[0])
            for frame in framen:
                rm = faceCrop+'face_'+str(frame)+'.jpg'
                try:
                    os.remove(rm)
                except:
                    print('No False Positive')