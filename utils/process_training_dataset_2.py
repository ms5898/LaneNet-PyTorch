import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset')
    return parser.parse_args()


def get_image_to_folders(json_label_path, gt_image_dir, gt_binary_dir, gt_instance_dir, src_dir):
    image_nums = len(os.listdir(gt_image_dir))
    with open(json_label_path, 'r') as file:
        for line_index, line in enumerate(file):
            info_dict = json.loads(line)

            raw_file = info_dict['raw_file']
            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            image_path = ops.join(src_dir, raw_file)
            image_name_new = '{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4))
            image_output_path = ops.join(ops.split(src_dir)[0], 'training', 'gt_image', image_name_new)
            binary_output_path = ops.join(ops.split(src_dir)[0], 'training', 'gt_binary_image', image_name_new)
            instance_output_path = ops.join(ops.split(src_dir)[0], 'training', 'gt_instance_image', image_name_new)

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    continue
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)

                cv2.polylines(dst_binary_image, lane_pts, isClosed=False, color=255, thickness=5)
                cv2.polylines(dst_instance_image, lane_pts, isClosed=False, color=lane_index * 50 + 20, thickness=5)

            cv2.imwrite(binary_output_path, dst_binary_image)
            cv2.imwrite(instance_output_path, dst_instance_image)
            cv2.imwrite(image_output_path, src_image)
        print('Process {:s} success'.format(json_label_path))


def gen_train_sample(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):
    os.makedirs('{:s}/txt_for_local'.format(ops.split(src_dir)[0]), exist_ok=True)
    with open('{:s}/txt_for_local/train.txt'.format(ops.split(src_dir)[0]), 'w') as file:
        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue
            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('Image set: {:s} broken'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')
    return


def split_train_txt(src_dir):
    train_file_path =  '{:s}/txt_for_local/train.txt'.format(ops.split(src_dir)[0])
    test_file_path = '{:s}/txt_for_local/test.txt'.format(ops.split(src_dir)[0])
    valid_file_path = '{:s}/txt_for_local/val.txt'.format(ops.split(src_dir)[0])
    with open(train_file_path, 'r') as file:
        data = file.readlines()
        train_data = data[0:int(len(data)*0.8)]
        test_data = data[int(len(data)*0.8): int(len(data)*0.9)]
        valid_data = data[int(len(data) * 0.9): -1]
    with open(train_file_path, 'w') as file:
        for d in train_data:
            file.write(d)
    with open(test_file_path, 'w') as file:
        for d in test_data:
            file.write(d)
    with open(valid_file_path, 'w') as file:
        for d in valid_data:
            file.write(d)


def process_tusimple_dataset(src_dir):
    traing_folder_path = ops.join(ops.split(src_dir)[0], 'training')
    os.makedirs(traing_folder_path, exist_ok=True)

    gt_image_dir = ops.join(traing_folder_path, 'gt_image')
    gt_binary_dir = ops.join(traing_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(traing_folder_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    for json_label_path in glob.glob('{:s}/*.json'.format(src_dir)):
        get_image_to_folders(json_label_path, gt_image_dir, gt_binary_dir, gt_instance_dir, src_dir)
    gen_train_sample(src_dir, gt_binary_dir, gt_instance_dir, gt_image_dir)
    split_train_txt(src_dir)


if __name__ == '__main__':
    args = init_args()
    process_tusimple_dataset(args.src_dir)