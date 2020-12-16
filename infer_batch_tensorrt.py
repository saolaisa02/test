from pdb import set_trace as pause
# from src.data_generator import DataGenerator
# from src.sampler import augment_sample, labels2output_map
# from src.utils import image_files_from_folder, show

# from src.sampler_train import augment_sample, labels2output_map
from src.utils_train import image_files_from_folder, show
from src.loss import loss
from src.label import readShapes
from src.keras_utils import save_model, load_model
from os import makedirs
from os.path import isfile, isdir, basename, splitext
from random import choice
import argparse
import cv2
import numpy as np
import sys
import os
import utils_dp

from dataset42 import DataGenerator #aug mới
from evaluator_batch_tensorrt import eval2

from glob import glob
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_network(modelpath, input_dim):
    model = load_model(modelpath)
    input_shape = (input_dim, input_dim, 3)

    # Fixed input size for training
    inputs = keras.layers.Input(shape=(input_dim, input_dim, 3))
    outputs = model(inputs)

    output_shape = tuple([s.value for s in outputs.shape[1:]])
    output_dim = output_shape[1]
    model_stride = input_dim / output_dim

    assert input_dim % output_dim == 0, \
        'The output resolution must be divisible by the input resolution'

    return model, model_stride, input_shape, output_shape
    
def read_data(name_file):
    
    data={}
    data['img_dir']=[]
    data['annotation']=[]

    f = open(name_file, "r")

    data_str = f.read()

    anns = data_str.split("image/bienso/")
    print(len(anns))
    for ann in anns:
        ann_lines = ann.split("\t")
        if(len(ann_lines) < 2):
            continue
        data['img_dir'].append(ann_lines[0]) 
        ann=[]
        anns2 = ann_lines[1].split('\n') 
        for idx in range(len(anns2)-1):
            ann.append(anns2[idx])
        data['annotation'].append(ann)
    
    print("Len ann: ", len(data['annotation']))
    print("Len img_dir: ", len(data['img_dir']))
    f.close()
    return data
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default='', type=str,
                        help='Path to previous model')
    args = parser.parse_args()  
    dim = 208
            

#             train_data = read_data("/data/anhnt2/utvm_data_split/new_1008/bienso_train_T10_clear.txt")
    test_data = read_data("/data/anhnt2/utvm_data_split/new_1008/bienso_test_T10_clear.txt")
#             train_data = read_data("/data/anhnt2/utvm_data_split/stanford/bienso_stanford_train.txt")
#     test_data = read_data("/data/anhnt2/utvm_data_split/stanford/bienso_stanford_test.txt")

    image_folder="/data/tagging_utvm/bienso/"

#             training_generator = DataGenerator(train_data['img_dir'], train_data['annotation'],image_folder, dim=dim, model_stride=model_stride ,batch_size=16)
    test_generator = DataGenerator(test_data['img_dir'], test_data['annotation'],image_folder, dim=288,val=True,batch_size=8)

    AP, R, P, AP_max, R_max, P_max,num_correct,num_correct_max = eval2(test_generator)
    print("AP_max: ", AP_max)
if __name__ == '__main__':
    main()

''' 

python infer_batch.py --model models/official_models/utvm_models_small_stanford/utvm-model-small-stanford_backup_max
python infer_batch.py --model final_models/IoU_05/utvm_models_resnet2/backup/utvm-model-privatedata-resnet2_backup_max-Copy1
python infer_batch.py --model final_models/IoU_05/utvm_models_resnet_cus/backup/utvm-model-stanford-resnet-cus_backup_max-Copy1
python infer_batch.py --model final_models/IoU_05/utvm_models_resnet_cus_stanford/utvm-model-stanford-resnet-cus_backup_max  


'''