# Counts successfully saved .pckl files in facetracks for each POI and language
import os, glob

id_list = glob.glob(os.path.join('facetracks', '*'))
id_list.sort()

for ids in id_list:
    print(f'{ids} -  deutsch: {len(glob.glob(os.path.join(ids, "deutsch", "**", "*.pckl"), recursive=True))}  - english: {len(glob.glob(os.path.join(ids, "english", "**", "*.pckl"), recursive=True))}')