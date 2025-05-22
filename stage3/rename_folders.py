import glob, os


print(glob.glob('identities/*'))

for folder in glob.glob('identities/*'):
    new_name = folder.replace(" ", "_")
    os.rename(folder, new_name)
    links = new_name

print(glob.glob('identities/*'))


inpDir = 'identities/*'
all_links = glob.glob(os.path.join(inpDir, '**', '*.mp4'), recursive=True)
all_links.sort()
#print(all_links.index("identities/George_Dzundza/deutsch/eKRXhZm0ciQ/Absolut_Kult!_-_George_Harrison_bei_Günther_Jauch_in_Na_Siehste_1988.mp4"))
print(len(all_links))