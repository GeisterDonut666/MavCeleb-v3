'''
M. Saad Saeed
18F-MS-CP-01
'''
import subprocess
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import os
import pickle
import cv2
import tensorflow as tf
import numpy as np
import glob
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

print('Libraries imported')


# NOTE: this code likely won't work in the environment presented in the documentation

def face_detect(inputVideo):
    PATH_TO_CKPT = 'model\\frozen_inference_graph_face.pb'
    MIN_CONF = 0.3
    cap = cv2.VideoCapture(inputVideo)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    scenefaces = []
    print('Networkds created')
    with detection_graph.as_default():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
          frame_num = 0
          print('Detecting Faces\n')
          while True:        
            ret, image = cap.read()
            if ret == 0:
                break
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            score = scores[0]
            scenefaces.append([])
            for index in range(0,len(score)):
              if score[index] > MIN_CONF:
                scenefaces[-1].append([frame_num, boxes[0][index].tolist(),score[index]])
            print('%s-%05d; %d dets' % ('Frame',frame_num,len(scenefaces[-1]))) 
            frame_num += 1  
          cap.release()
    print('Faces detected\n')
    return scenefaces

def scene_detect(inputVideo):
    video_manager = VideoManager([inputVideo])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]
    print('Scenes Detected %d'%(len(scene_list)))
    return scene_list


def bb_intersection_over_union(boxA, boxB):
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  interArea = max(0, xB - xA) * max(0, yB - yA)
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou

def track_shot(scenefaces):
    iouThres  = 0.5
    numFail   = 3
    minSize   = 0.05
    tracks    = []
    min_track = 100
    while True:
        track     = []
        for faces in scenefaces:
            for face in faces:
                if track == []:
                    track.append(face)
                    faces.remove(face)
                elif face[0] - track[-1][0] <= numFail:
                    iou = bb_intersection_over_union(face[1], track[-1][1])
                    if iou > iouThres:
                        track.append(face)
                        faces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > min_track:
            framenum    = np.array([ f[0] for f in track ])
            bboxes    = np.array([np.array(f[1]) for f in track])
            frame_i   = np.arange(framenum[0],framenum[-1]+1)
            bboxes_i    = []
            for ij in range(0,4):
                interpfn  = interp1d(framenum, bboxes[:,ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i  = np.stack(bboxes_i, axis=1)
            
            if np.mean(bboxes_i[:,3]-bboxes_i[:,1]) > minSize:
                tracks.append([frame_i,bboxes_i])
    return tracks

def crop_video(inputVideo,track,cropfile,tmpDir):
    crop_scale = 0.5
    cap = cv2.VideoCapture(inputVideo)  
    cap.set(1,track[0][0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, cap.get(5), (224,224))
    fw = cap.get(3)
    fh = cap.get(4)
    dets = [[], [], []]
    for det in track[1]:
        dets[0].append(((det[3]-det[1])*fw+(det[2]-det[0])*fh)/4)
        dets[1].append((det[1]+det[3])*fw/2) 
        dets[2].append((det[0]+det[2])*fh/2) 
    dets[0] = signal.medfilt(dets[0],kernel_size=5)   
    dets[1] = signal.medfilt(dets[1],kernel_size=5)
    dets[2] = signal.medfilt(dets[2],kernel_size=7)
    for det in zip(*dets):
        cs  = crop_scale
        bs  = det[0] 
        bsi = int(bs*(1+2*cs))
        ret, frame = cap.read()  
        frame = np.pad(frame,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(0,0))
        my  = det[2]+bsi 
        mx  = det[1]+bsi 
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]    
        vOut.write(cv2.resize(face,(224,224)))
    audiotmp  = tmpDir+'\\audio.wav'
    audiostart  = track[0][0]/cap.get(5)
    audioend  = (track[0][-1]+1)/cap.get(5)
    cap.release()
    vOut.release()
    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f %s" % (os.path.join(tmpDir,'video.avi'),audiostart,audioend,audiotmp)) #-async 1 
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        print('error')
    sample_rate, audio = wavfile.read(audiotmp)
    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile)) 
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        print('error')
    print('Written %s'%cropfile)
    os.remove(cropfile+'t.avi')
    return [track,dets]


def run_facetrack(inpDir, tmpDir, outDir):
    min_track = 100
    for ids in glob.glob(inpDir):
        for ty in glob.glob(ids+'\\Urdu'):
            for links in glob.glob(ty+'\\*\\*'):
                tp = links.split('.')[0].split('\\')[1:4]
                tp = outDir+tp[0]+'\\'+tp[1]+'\\'+tp[2]+'\\'
                if not os.path.exists(tp):
                    os.makedirs(tp)
                try:
                    subprocess.call("ffmpeg -y -i %s -qscale:v 4 -r 25 %s" % (links,tmpDir+'\\video.avi'),shell=True,stdout=None)
                    inputVideo = tmpDir+'\\video.avi'
                    print(inputVideo+ ' Converted to AVI')
                    alltracks = []
                    vidtracks = []
                    faces = []
                    scene = []
                    faces = face_detect(inputVideo)
                    scene = scene_detect(inputVideo)
                    for shot in scene:
                        print(shot[1].frame_num - shot[0].frame_num)
                        if shot[1].frame_num - shot[0].frame_num >= min_track :
                            alltracks.extend(track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
                    for ii, track in enumerate(alltracks):
                        vidtracks.append(crop_video(inputVideo,track,tp+'\\%05d'%ii,tmpDir))
                    with open(tp+'tracks.pckl', 'wb') as file:
                        pickle.dump(vidtracks, file)
                except:
                    print('error in:',tp)
                    with open('viderrU.txt','a+') as file:
                        file.write(tp+'\n')
                             
tmpDir = 'tmpUrdu\\'
outDir = 'facetracks\\'
inpDir = 'data\\*'

if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)

run_facetrack(inpDir, tmpDir, outDir)