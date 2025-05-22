"""
M. Saad Saeed
18F-MS-CP-01
"""


#import glob
#import os

#orLen = 268
#alltxt = []
#count=0
#for ids in glob.glob('facetracks\\*\\*\\*'):
#    tmptxt = []
#    su = 0
#    for links in glob.glob(ids+'\\*.txt'):
#        with open(links, 'r+') as file:
#            for i,dat in enumerate(file):
#                if i==3:
#                    su = su+float(dat.split(':')[1])
#                    tmptxt.append([links,dat.split(':')[1]])
#        count+=1
#    if len(tmptxt)!=0:
#        avg = su/len(tmptxt)
#    else:
#        avg = 0
#    print(links)
#    with open('avgs20.txt','a+') as file:
#        file.write(ids+'\t')
#        file.write(str(avg)+'\n')
#    alltxt.append([ids,avg])
#    
#ii = 0
#tobeDel = []
#for ids in glob.glob('facetracks\\*\\*\\*'):
#    avg = alltxt[ii][1]
#    for links in glob.glob(ids+'\\*.txt'):
#        if avg !=0:
#            with open(links,'r+') as file:
#                for i,dat in enumerate(file):
#                    if i==3:
#                        sc = float(dat.split(':')[1])
#                        if avg < 3:
#                            if sc < avg:
#                                tobeDel.append([links,sc,avg])
#                        if sc < avg:
#                            if avg > 3 and sc < 3:
#                                tobeDel.append([links,sc,avg])
#    ii+=1
##    
#for lines in tobeDel:
#    with open('tobeDel20.txt','a+') as file:
#        file.write(lines[0]+'\t'+str(lines[1])+'\t'+str(lines[2])+'\n')

#count=0
#tobedel = []
#with open('tobeDel20.txt','r+') as file:
#    for dat in file:
#        tobedel.append(dat)
#        txt = dat.split('.txt')[0]+'.txt'
#        vid = dat.split('.txt')[0][0:-4]+'%05d.avi'%(int(dat.split('\\')[4].split('.txt')[0]))
#        try:
#            os.remove(txt)
#            os.remove(vid)
#        except:
#            print('Already Deleted')
#            count+=1
#
#
#
#for lines in tobeDel:
#    txt = lines[0]
#    vid = lines[0].split('.txt')
#    print('')
#    os.remove(lines[0])
#
#    
#
#
#    
#
#
#
#de=0
#if len(alltxt) == orLen:
#    for i,dat in enumerate(alltxt):
#        avg = dat[1]
#        for moredat in dat[0]:
#            with open(moredat[0],'r+') as file:
#                for i,score in enumerate(file):
#                    if i==3:
#                        tmpScore = score.split(':')[1]
#                        if avg>=tmpScore:
#                            de+=1