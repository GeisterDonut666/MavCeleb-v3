"""
M. Saad Saeed
18F-MS-CP-01
"""


import glob, subprocess, os
import shlex
import shutil
outDir = 'out'  # I don't think this is used, actually
tmpDir = 'tmpdir'
if not os.path.exists(outDir):
    os.mkdir(outDir)
if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)

count=0
noAudio = []
id_list = glob.glob(os.path.join('identities', '*'))
id_list.sort()



for ids in id_list:
    for ty in glob.glob(os.path.join(ids, '*')):
        for links in glob.glob(os.path.join(ty, '*', '*.mp4')):
            count+=1
            print(links)
            #command = "ffprobe -i %s -show_streams -select_streams a -loglevel error" % (links)        # for Windows (single string expexted as input).
            command = ["ffprobe", "-i", links, "-show_streams", "-select_streams", "a", "-loglevel", "error"]                   # for linux                                                                          # for Linux (list of arguments is needed)
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            out, err = p.communicate()

            d = len(out)
            if d <= 10:
                os.remove(links)



with open('noAudio.txt','w+') as file:
    for txt in noAudio:
       file.write(txt+'\n')
        

with open('noAudio.txt','r+') as file:
    for txt in file:
        cop = txt.split('\\')[2]
        noAudio.append(txt)
        shutil.copytree('111\\'+cop,txt.split('\n')[0])
        print('Copying: ',txt)
        
#for down in glob.glob('')

for ids in id_list:
    print(f"{ids} -  deutsch: {len(glob.glob(os.path.join(ids, 'deutsch', '**', '*.mp4'), recursive=True))}  - english: {len(glob.glob(os.path.join(ids, 'english', '**', '*.mp4'), recursive=True))}")