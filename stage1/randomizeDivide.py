"""
M. Saad Saeed
18F-MS-CP-01
"""
import random

identity = []
stud = []
with open('newCelebList.txt','r+') as file:
    for names in file:
        identity.append(names)

with open('students.txt','r+') as file:
    for names in file:
        sd = names.split('-')
        stud.append([sd[0]+'-'+sd[1]+'-'+sd[2],sd[3]])



random.shuffle(identity)
celebs  = []
tempCeleb = []
prev = 0
for assign in stud:
    for s,i in zip(assign[0:],assign[1:]):
        curr = prev+int(i)
        tempCeleb = identity[prev:curr]
        celebs.append(tempCeleb)
        prev = curr
        with open('16-CP-yy\\'+s+'.txt', 'w+') as file:
            for cid in tempCeleb:
                file.write(cid)
#    for i in range(assign):
#        with open('16-CP-yy\\'+)
#
#i=0
#for sid in stud:
#    print(sid)
#    tempCeleb = identity[i*idbatch:idbatch*(i+1)]
#    with open('final\\'+sid.split('\n')[0]+'.txt','w+') as file:
#        for cid in tempCeleb:
#            file.write(cid)
#    i+=1
#
#identity =[]
#for sid in stud:
#    with open('final\\'+sid.split('\n')[0]+'.txt','r+') as file:
#        for txt in file:
#            identity.append(txt)