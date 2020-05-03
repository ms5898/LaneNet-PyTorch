import os.path as ops
import numpy as np
import torch
import cv2
import time
import tqdm
import os
from sklearn.cluster import MeanShift, estimate_bandwidth


def gray_to_rgb_emb(gray_img):
    """
    :param gray_img: torch tensor 256 x 512
    :return: numpy array 256 x 512
    """
    H, W = gray_img.shape
    element = torch.unique(gray_img).numpy()
    rbg_emb = np.zeros((H, W, 3))
    color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 215, 0], [0, 255, 255]]
    for i in range(len(element)):
        rbg_emb[gray_img == element[i]] = color[i]
    return rbg_emb/255


def process_instance_embedding(instance_embedding, binary_img, distance=1, lane_num=5):
    embedding = instance_embedding[0].detach().numpy().transpose(1, 2, 0)
    cluster_result = np.zeros(binary_img.shape, dtype=np.int32)
    cluster_list = embedding[binary_img > 0]
    mean_shift = MeanShift(bandwidth=distance, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(cluster_list)
    labels = mean_shift.labels_

    cluster_result[binary_img > 0] = labels + 1
    cluster_result[cluster_result > lane_num] = 0
    for idx in np.unique(cluster_result):
        if len(cluster_result[cluster_result == idx]) < 15:
            cluster_result[cluster_result == idx] = 0

    H, W = binary_img.shape
    rbg_emb = np.zeros((H, W, 3))
    color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 215, 0], [0, 255, 255]]
    element = np.unique(cluster_result)
    for i in range(len(element)):
        rbg_emb[cluster_result == element[i]] = color[i]

    return rbg_emb / 255, cluster_result


def video_to_clips(video_name):
    test_video_dir = ops.split(video_name)[0]
    outimg_dir = ops.join(test_video_dir, 'clips')
    if ops.exists(outimg_dir):
        print('Data already exist in {}'.format(outimg_dir))
        return
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []

    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1

    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i + 1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)
    print('finish process and save in {}'.format(outimg_dir))




