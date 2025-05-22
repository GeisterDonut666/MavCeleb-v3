import os, glob

names = []
with open(glob.glob(os.path.join('**', 'candidate_list'), recursive=True)[0], 'r') as f:
    for name in f.readlines():
        names.append(name)
names.sort()

with open(glob.glob(os.path.join('**', 'candidate_list'), recursive=True)[0], 'w') as f:
        f.writelines(names)