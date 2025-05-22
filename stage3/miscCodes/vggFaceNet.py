import cv2
import glob

mylines = []                             # Declare an empty list named mylines.
with open ('id0001.txt', 'rt') as myfile: # Open lorem.txt for reading text data.
    for myline in myfile:                # For each line, stored as myline,
        mylines.append(myline)     
idx = mylines[0].split(': id')[1]
reference = mylines[1].split(': ')[1]
x=[]
y=[]
w=[]
h=[]
framen=[]
cords = mylines[4:len(mylines)]
for inp in range(len(cords)):
    temp = cords[inp]
    temp = temp.split('\t')[0:5]
    framen.append(temp[0])
    x.append(temp[1])
    y.append(temp[2])
    w.append(temp[3])
    h.append(temp[4])
images = glob.glob('frames\\*.jpg')  

count = 0;
for cap in images:
    im = cv2.imread(cap)
    im = im[int(x[count]):int(x[count])+int(w[count]),int(y[count]):int(y[count])+int(h[count])]
    cv2.imshow('im',im[int(y[count]):int(y[count])+int(h[count]),int(x[count]):int(x[count])+int(w[count])])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    count +=1;
    
    
#frame1 = cv2.imread('frames\\frame00001.jpg')
#frame9 = cv2.imread('frames\\frame00011.jpg')
#im  = frame1[int(x[count]):int(x[count])+int(w[count]),int(y[count]):int(y[count])+int(h[count])]
#cv2.imshow('im',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow('im',frame9)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#frame9[int(x[count]):int(x[count])+int(w[count]),int(y[count]):int(y[count])+int(h[count])] = im
#cv2.imshow('im',frame9)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
