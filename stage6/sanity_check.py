import glob, os

id_paths = glob.glob(os.path.join('cleanVid', '*')); id_paths.sort()
print(id_paths)

for id in id_paths:
    for typ in glob.glob(os.path.join(id, '*')):
        i = len(glob.glob(os.path.join(typ, '*')))
        print(f'{typ}: {i}')

