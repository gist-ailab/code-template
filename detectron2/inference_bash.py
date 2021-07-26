import numpy as np
import json
import subprocess
import os
from multiprocessing import Process
import argparse
from glob import glob
import warnings
warnings.filterwarnings('ignore')


if __name__=='__main__':
    # Setup Configuration for Each Experiments
    num_per_gpu = 1
    gpus = ['0']

    config_path = './configs/faster_rcnn_KDN_inference.yaml'
    batch_size = 48

    # Input Folder
    in_base = '/home/sung/dataset/KDN/test_videos/enhancement'

    # Save Folder for Each Results
    inference_folder = './inference_result_enhancement'
    visualize_folder = './visual_result_enhancement'

    clip_list = os.listdir(in_base)

    frame = 48
    nms_threshold = 0.3
    confidence_threshold = 0.8
    target = None

    comb_list = []
    ix = 0
    for clip_name in clip_list:
        comb_list.append([os.path.join(in_base, clip_name), ix])
        ix += 1

    comb_list = comb_list * num_per_gpu
    comb_list = [comb + [index] for index, comb in enumerate(comb_list)]

    arr = np.array_split(comb_list, len(gpus))
    arr_dict = {}
    for ix in range(len(gpus)):
        arr_dict[ix] = arr[ix]

    def tr_gpu(comb, ix):
        comb = comb[ix]
        for i, comb_ix in enumerate(comb):
            gpu = gpus[ix]

            # Get Result !
            script = 'python ./inference.py --config_file %s --batch_size %d --output %s \
                                             --input %s --nms_threshold %.2f --gpu %s' %(config_path, batch_size, inference_folder, str(comb_ix[0]), nms_threshold, gpu)

            subprocess.call(script, shell=True)

            # Visualize Result !
            clip_name = str(comb_ix[0]).split('/')[-1]
            result_path = os.path.join(inference_folder, 'result_%s.pkl' %clip_name)
            vis_folder = os.path.join(visualize_folder, 'vis_%s' %clip_name)

            if target is None:
                script = 'python ./utils/visualize.py --result_path %s \
                                                      --save_folder %s \
                                                      --confidence_threshold %.2f' %(result_path, vis_folder, confidence_threshold)
            else:
                script = 'python ./utils/visualize.py --result_path %s \
                                                      --save_folder %s \
                                                      --target_class %s \
                                                      --confidence_threshold %.2f' %(result_path, vis_folder, target_class, confidence_threshold)
            subprocess.call(script, shell=True)

            # Convert Images to Videos
            video_path = os.path.join(visualize_folder, 'demo_%s.avi' %clip_name)

            script = 'python ./utils/video_gen.py --input_dir %s \
                                                  --video_path %s \
                                                  --frame %d' %(vis_folder, video_path, frame)
            subprocess.call(script, shell=True)

    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)