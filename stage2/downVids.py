"""
M. Saad Saeed
18F-MS-CP-01
"""

# from pytube import YouTube    - originally used this library, but it is now deprecated
from pytubefix import YouTube
import glob
import shutil
import os

links_sorted = glob.glob(os.path.join('identities', '*'))
links_sorted.sort()
fuckups = 0

for idList in links_sorted:
    print(idList)
    with open('vidErr.txt','a+') as file:
        file.write(idList+'\n')
    for linkstxt in glob.glob(os.path.join(idList, '*.txt')):

        try:
            shutil.rmtree(linkstxt.split('.')[0])
        except:
            print('Already Deleted')
        os.mkdir(linkstxt.split('.')[0])
        links = []
        with open(linkstxt,'r+') as file:
            for link in file:
                links.append(link)
            for index in range(len(links)):
                try:
                    link = links[index]
                    # Don't download already existing videos again
                    #if len(glob.glob(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))) > 0:
                    #    continue

                    try:
                        os.mkdir(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                    except:
                        shutil.rmtree(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                        try:
                            os.mkdir(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                        except:
                            print('Error')
                    print(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                    try:
                        yt = YouTube(link)
                        index+=1
                        print(index)
                    except:
                        with open('vidErr.txt','a+') as file:
                            file.write(link)
                        index+=1
                        if index >= len(links):
                            break
                        link = links[index]
                        print('Video Secured')
                        try:
                            yt = YouTube(link)
                        except:
                            print('Error: Check Manually')
                    try:
                        stream = yt.streams.filter(file_extension='mp4',res='720p').first()
                    except:
                        print('Error: Check Manually')
                    try:
                        stream.download(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                    except:
                        stream = yt.streams.filter(file_extension='mp4',res='480p').first()
                        try:
                            stream.download(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                        except:
                            stream = yt.streams.filter(file_extension='mp4',res='360p').first()
                            try:
                                stream.download(os.path.join(linkstxt.split('.')[0], link.split('=')[1].split('\n')[0]))
                            except:
                                print('User check manually')
                except:
                    fuckups += 1
                    print(f"FUCKUP COUNTER: {fuckups}")
