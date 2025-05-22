import glob, os

print(glob.glob('facetracks/*/*/*.pckl'))


for folder in glob.glob('facetracks/*/*/*'):
    for file in glob.glob(folder + '/*.pckl'):
        os.rename(file, file.replace('track.pckl', 'tracks.pckl'))

'''
print(glob.glob('facetracks/*'))

inpDir = 'identities/*'
all_links = glob.glob(os.path.join(inpDir, '**', '*.mp4'), recursive=True)
all_links.sort()
print(all_links.index("identities/George_Dzundza/deutsch/eKRXhZm0ciQ/Absolut_Kult!_-_George_Harrison_bei_Günther_Jauch_in_Na_Siehste_1988.mp4"))
print(len(all_links))
'''
