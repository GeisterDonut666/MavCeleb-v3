'''
M. Saad Saeed
18F-MS-CP-01
'''

import os
import pickle
import numpy as np
import glob
from scipy import signal 


def run_sync():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from SyncNetInstance import SyncNetInstance
    import argparse
    parser = argparse.ArgumentParser(description = "SyncNet");
    parser.add_argument('--initial_model', type=str, default="model/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='64', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--data_dir', type=str, default='', help='')
    parser.add_argument('--videofile', type=str, default='', help='') 
    opt = parser.parse_args()
    setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'tmpE'))
    s = SyncNetInstance()
    s.loadParameters(modDir);
    print("Model %s loaded."%modDir)
    all_links = glob.glob('facetracks/*', recursive=True)
    all_links.sort()
    print(len(all_links))
    for ids in all_links[::]:
        print('------------------------------')
        print(ids)
        try:
            print(all_links.index(ids))
        except ValueError:
            pass
        print(glob.glob(ids + '/deutsch'))

        for ty in glob.glob(ids+'/deutsch'):
            for links in glob.glob(ty+'/*'):
                print(links)
                try:
                    with open(links+'/tracks.pckl','rb') as file:
                       vidtracks = pickle.load(file,encoding='latin1')
                except:
                    continue
                dists = []
                offsets = []
                confs = []
                try:
                    for ii, track in enumerate(vidtracks):
                        offset, conf, dist = s.evaluate(opt,videofile=os.path.join(links,'%05d.avi'%ii))
                        offsets.append(offset)
                        dists.append(dist)
                        confs.append(conf)
                    with open(links+'/dists.pckl','wb') as file:
                        pickle.dump(dists, file)
                    for ii in range(len(vidtracks)):
                        with open(links+'/%04d.txt'%(ii),'w+') as file:
                            file.write('Reference:\t%s\n'%(links.split('/')[3]))
                            file.write('Id:\t%s\n'%(ids.split('/')[1]))
                            file.write('Offset:\t\t%d\nASD Conf.:\t%0.3f\n\n'%(offsets[ii],confs[ii]))
                            file.write('Frame\tX\tY\tW\tH\n\n')
                    faces = [ [] for ii in range(100000)]
                    for ii, track in enumerate(vidtracks):
                    	mean_dists =  np.mean(np.stack(dists[ii],1),1)
                    	minidx = np.argmin(mean_dists,0)
                    	fdist   	= np.stack([dist[minidx] for dist in dists[ii]])
                    	fdist   	= np.pad(fdist, (3,3), 'constant', constant_values=10)
                    	fconf   = np.median(mean_dists) - fdist
                    	fconfm  = signal.medfilt(fconf,kernel_size=9)
                    	for ij, frame in enumerate(track[0][0].tolist()) :
                    		faces[frame].append([ii, fconfm[ij], track[1][0][ij], track[1][1][ij], track[1][2][ij]])
                    for ii,track in enumerate(vidtracks):
                        track = track[0][0]
                        for i in range(len(track)):
                            with open(links+'/%04d.txt'%(ii),'a+') as fil:
                                fil.write('%06d\t%d\t%d\t%d\t%d\n'
                                          %(track[i],faces[track[i]][0][3]-faces[track[i]][0][2],
                                faces[track[i]][0][4]-faces[track[i]][0][2],
                                faces[track[i]][0][3]+faces[track[i]][0][2]-(faces[track[i]][0][3]-faces[track[i]][0][2]),
                                faces[track[i]][0][4]+faces[track[i]][0][2]-(faces[track[i]][0][4]-faces[track[i]][0][2])))
                except:
                    with open('errsyncE.txt','a+') as file:
                        file.write(links)


tmpDir = 'tmpE/'
modDir = 'model/syncnet_v2.model'
if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)
run_sync()