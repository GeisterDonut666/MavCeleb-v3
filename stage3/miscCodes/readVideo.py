"""
M. Saad Saeed
18F-MS-CP-01
"""
import cv2
import glob
from multiFaceDetection import createNetwork, faceDetect

parent = 'dataSet\\'
if 'rnet' not in globals():
    print('Creating Network')
    pnet, onet, rnet = createNetwork()
identityList = glob.glob(parent+'*')
for identity in identityList:
    fold1 = identity.split('\\')[1]
    with open(fold1+".txt","w+") as file:
            file.write("Identity: %s" % (fold1))
            file.write("\n")
    idx = identity+'\\*'
    linkList = glob.glob(idx)
    for video in linkList:
        fold2  = video.split("\\")[2]
        with open(fold1+".txt","a+") as file:
            file.write("Reference: %s \n\n" % (fold2))
            file.write("Frames\tX\tY\tW\tH\tAcc\n")
        vidx =  video+'\\*'
        vidx = 've.mp4'
        vidList = glob.glob(vidx)
        cap = cv2.VideoCapture(vidList[0])
        i =0;
        while(cap.isOpened()):
#            pnet, onet, rnet = createNetwork()
            ret, frame = cap.read()
            if ret:
                
#                if i>1:
#                    break
#                faceDetect(frame,i,pnet,onet,rnet,fold1)
                cv2.imwrite("frames\\frame{:05d}.jpg".format(i),frame)
                i= i+1
                print(i)
            else:
                print('Done')
                cap.release()
                
# [3]:
# 868 118 1059 367
#            x1 = b[0], x2 = [bb3], y1 =   bb[2], y2 = bb[4]
#im = cv2.imread('abc.jpg')
#cropped = im[152:152+182,331:331+148]
##cropped = im[bb[1]:bb[3]-bb[1]+bb[1],bb[0]:bb[2]-bb[0]+bb[0]]
#cv2.imshow('asdf',cropped)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
                
# I[3]:


#from shot_detection import histogram             
#
#histograms = histogram(im)




