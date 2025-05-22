"""
M. Saad Saeed
18F-MS-CP-01
"""
import numpy as np
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
images = images[1:len(images)] 
#cr = im[int(x[0]):int(x[0])+int(w[0]),int(y[0]):int(y[0])+int(h[0])]

feature_params = dict( maxCorners = 500,   # How many pts. to locate
                       qualityLevel = 0.1,  # b/w 0 & 1, min. quality below which everyone is rejected
                       minDistance = 0.4,   # Min eucledian distance b/w corners detected
                       blockSize = 3 ) # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (3,3),  # size of the search window at each pyramid level
                  maxLevel = 0,   #  0, pyramids are not used (single level), if set to 1, two levels are used, and so on
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.9))


# Take first frame and find corners in it
size = (160,160)
im = cv2.imread(images[0])
old_frame = im[int(x[0]):int(x[0])+int(w[0]),int(y[0]):int(y[0])+int(h[0])]
old_frame = cv2.resize(old_frame,size, interpolation= cv2.INTER_CUBIC)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  #use goodFeaturesToTrack to find the location of the good corner.

# Create a mask image for drawing purposes filed with zeros
mask = np.zeros_like(old_frame)

k = 0
count = 1  # for the frame count
n = 1  # Frames refresh rate for feature generation

for cap in images:
    im = cv2.imread(cap)
    frame = im[int(x[count]):int(x[count])+int(w[count]),int(y[count]):int(y[count])+int(h[count])]
    frame = cv2.resize(frame,size, interpolation= cv2.INTER_CUBIC)
    if frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if count%n == 0:  # Refresh the tracking features after every 50 frames
        cv2.imwrite('tracked/track{0:05d}.jpg'.format(k), frame)
        k += 1
        im = cv2.imread(cap)
        old_frame = im[int(x[count]):int(x[count])+int(w[count]),int(y[count]):int(y[count])+int(h[count])]
        old_frame = cv2.resize(old_frame,size,interpolation= cv2.INTER_CUBIC)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    print('good old',np.sum(p0))
    print('good new',np.sum(p1))
    
    
    if np.sum(p1-p0)>10:
        print('change detected at frame no %d',count)
    
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #tmp new value
        c,d = old.ravel() #tmp old value
        #draws a line connecting the old point with the new point
        mask = cv2.line(mask, (a,b),(c,d), (0,255,0), 1)
        #draws the new point
        frame = cv2.circle(frame,(a,b),2,(0,0,255), -1)
    
    img = cv2.add(frame,mask)
    img = cv2.resize(img,(int(h[count]),int(w[count])))
    im[int(x[count]):int(x[count])+int(w[count]),int(y[count]):int(y[count])+int(h[count])] = img
    cv2.rectangle(im,(int(y[count]),int(x[count])),(int(y[count])+int(h[count]),int(x[count])+int(w[count])),(0,0,255),2)
#    out.write(img)
    cv2.imshow('frame',im)
    key = cv2.waitKey(30) & 0xff

    #Show the Output
    if key == 27:
        cv2.imshow('', im)
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    count += 1

# release and destroy all windows
cv2.destroyAllWindows()
#cap.release()