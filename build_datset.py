import torch
from torchvision.datasets import Kitti
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import shutil

import os
import csv

from calibration import *

path_dataset = os.path.normpath('datasets/')



def create_dataset(labels_dir,img_dir, new_labels_to, new_images_to):
    img_shape = (375, 1242)
    reverse_labels_dict = {0: 'Pedestrian',
                            1: 'Cyclist',
                            2: 'Car'}
    
    labels_dict = { 'Pedestrian': 0,
                    'Cyclist': 1,
                    'Car': 2}
    

    for file_ in os.listdir(labels_dir):
        tmp_path = os.path.join(labels_dir, file_)
        new_file = os.path.join(new_labels_to, file_)
        new_lines = []

        # Read the data from the labels files
        with open(tmp_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                elements = [line[e] for e in [0, 4, 5, 6, 7]]

                # Normalize bbox coordinates
                x = (float(elements[1]) + float(elements[3])) / (2 * img_shape[1])
                y = (float(elements[2]) + float(elements[4])) / (2 * img_shape[0])
                w = (float(elements[3]) - float(elements[1])) / img_shape[1]
                h = (float(elements[4]) - float(elements[2])) / img_shape[0]
                
                elements[1] = x
                elements[2] = y
                elements[3] = w
                elements[4] = h

                # Translate type to numbers and convert types of cars and people to standard
                type_ = elements[0]
                if type_ in ['Pedestrian', 'Person_sitting']:
                    elements[0] = labels_dict['Pedestrian']
                elif type_ == 'Cyclist':
                    elements[0] = labels_dict['Cyclist']
                elif type_ in ['Van', 'Truck', 'Car']:
                    elements[0] = labels_dict['Car']

                # Add the translated elements for the new file                
                if elements[0] in labels_dict.values():
                    elements = list(map(str, elements))
                    tmp_line = ' '.join(elements)
                    new_lines.append(tmp_line)

        # Write the new translated classes to the new file
        with open(new_file, 'w') as f:
            for line in new_lines:
                f.write(f'{line}\n')


    for file_ in os.listdir(img_dir):
        img_path = os.path.join(img_dir,file_)
        copy_path = os.path.join(new_images_to,file_)

        shutil.copyfile(img_path, copy_path)


def create_dataset2(labels_dir,img_dir, new_labels_to, new_images_to, addon_name):
    reverse_labels_dict = {0: 'Pedestrian',
                            1: 'Cyclist',
                            2: 'Car'}
    labels_dict = { 'Pedestrian': 0,
                    'Cyclist': 1,
                    'Car': 2}    



    df_temp = pd.read_csv(labels_dir, sep=' ', header=None)
    df_labels = pd.DataFrame(data={
    'frame': df_temp[0],
    'id': df_temp[1],
    'label': df_temp[2],
    'truncated': df_temp[3],
    'occluded': df_temp[4],
    'alpha': df_temp[5],
    'bbox': list(df_temp[list(range(6, 10))].values),
    'dimensions': df_temp[list(range(10, 13))].values.tolist(),
    'location': df_temp[list(range(13, 16))].values.tolist(),
    'rotation_y': df_temp[16],
    # 'score': df_temp[16],
    })


    df_labels['label'] = df_labels['label'].replace(labels_dict)



    for idx, img_file in enumerate(os.listdir(img_dir)):
    
        path_to_file = os.path.join(img_dir, img_file)
        # print(path_to_file)
        img_save_path = os.path.join(new_images_to, addon_name + img_file)
        label_save_path = os.path.join(new_labels_to, addon_name + img_file.split('.')[0] + '.txt')

        # Block to rectify and save image into custom dataset
        img_bgr = cv2.imread(path_to_file)
        height, width = img_bgr.shape[:2]
        # img_rectified = rectify_frame(img_bgr, left_cam_params)
        cv2.imwrite(img_save_path, img_bgr)

        # Block to save image labels into custom dataset
        img_labels = df_labels[df_labels['frame'] == idx].copy()
        img_labels = img_labels[['label', 'bbox']]
        x_col, y_col, w_col, h_col = [], [], [], []
        for i, (x, y, w, h) in enumerate(img_labels['bbox']):
            x_norm = ((w + x) / 2) / width
            w_norm = (w - x) / width
            y_norm = ((h + y) / 2) / height
            h_norm = (h - y) / height

            x_col.append(x_norm)
            y_col.append(y_norm)
            w_col.append(w_norm)
            h_col.append(h_norm)
            # img_labels['bbox'].iloc[i] = '{} {} {} {}'.format(x_norm, y_norm, w_norm, h_norm)
        
        img_labels = img_labels.assign(x_norm = x_col,
                                                    y_norm = y_col,
                                                    w_norm = w_col,
                                                    h_norm = h_col) 
        img_labels.drop('bbox', axis=1, inplace=True)
        img_labels.to_csv(label_save_path, sep=' ', header=False, index=False)




def create_dataset_test(labels_dir,img_dir, new_labels_to, new_images_to, addon_name):
    reverse_labels_dict = {0: 'Pedestrian',
                            1: 'Cyclist',
                            2: 'Car'}
    labels_dict = { 'Pedestrian': 0,
                    'Cyclist': 1,
                    'Car': 2}    





    for idx, img_file in enumerate(os.listdir(img_dir)):
    
        path_to_file = os.path.join(img_dir, img_file)
        # print(path_to_file)
        img_save_path = os.path.join(new_images_to, addon_name + img_file)
        label_save_path = os.path.join(new_labels_to, addon_name + img_file.split('.')[0] + '.txt')

        # Block to rectify and save image into custom dataset
        img_bgr = cv2.imread(path_to_file)
        height, width = img_bgr.shape[:2]
        # img_rectified = rectify_frame(img_bgr, left_cam_params)
        cv2.imwrite(img_save_path, img_bgr)



# makes the directory structure
if not os.path.exists(path_dataset):
    os.mkdir(path_dataset)
    os.mkdir('datasets/custom/')
    os.mkdir('datasets/custom/training/')
    os.mkdir('datasets/custom/training/images/')
    os.mkdir('datasets/custom/training/labels/')
    os.mkdir('datasets/custom/val/')
    os.mkdir('datasets/custom/val/images/')
    os.mkdir('datasets/custom/val/labels/')
    os.mkdir('datasets/custom/testing/')
    os.mkdir('datasets/custom/testing/images/')
    os.mkdir('datasets/custom/testing/labels/')


# Downloads the KITTI dataset to the folder 
path_dataset = r"./raw_data"
kitti_dataset = Kitti(path_dataset, download=True)


# Create the different datasets
root_dir = os.getcwd()


# Training set 
training_labels_dir = os.path.join(root_dir,r"raw_data\Kitti\raw\training\label_2")
training_new_labels_to = os.path.join(root_dir, r"datasets\custom\training\labels")
training_img_dir = os.path.join(root_dir, r"raw_data\Kitti\raw\training\image_2")
training_new_images_to = os.path.join(root_dir, r"datasets\custom\training\images")

if not os.listdir(training_new_images_to):
    if not os.listdir(training_new_labels_to):
        create_dataset(training_labels_dir,training_img_dir, training_new_labels_to, training_new_images_to)



# Validation set 
val_labels_dir1 = os.path.join(root_dir,r"raw_data\rectified\seq_01\labels.txt")
val_labels_dir2 = os.path.join(root_dir,r"raw_data\rectified\seq_02\labels.txt")
val_new_labels_to = os.path.join(root_dir, r"datasets\custom\val\labels")
val_img_dir1 = os.path.join(root_dir, r"raw_data\rectified\seq_01\image_02\data")
val_img_dir2 = os.path.join(root_dir, r"raw_data\rectified\seq_02\image_02\data")
val_new_images_to = os.path.join(root_dir, r"datasets\custom\val\images")

if not os.listdir(val_new_images_to):
    if not os.listdir(val_new_labels_to):
        create_dataset2(val_labels_dir1,val_img_dir1, val_new_labels_to, val_new_images_to,"seq01_")
        create_dataset2(val_labels_dir2,val_img_dir2, val_new_labels_to, val_new_images_to,"seq02_")


# Test set
test_labels_dir = os.path.join(root_dir,r"raw_data\rectified\seq_03\labels.txt")
test_new_labels_to = os.path.join(root_dir, r"datasets\custom\testing\labels")
test_img_dir = os.path.join(root_dir, r"raw_data\rectified\seq_03\image_02\data")
test_new_images_to = os.path.join(root_dir, r"datasets\custom\testing\images")

if not os.listdir(test_new_images_to):
    if not os.listdir(test_new_labels_to):
        create_dataset_test(test_labels_dir,test_img_dir, test_new_labels_to, test_new_images_to,"seq03_")