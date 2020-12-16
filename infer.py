from pdb import set_trace as pause
# from src.data_generator import DataGenerator
# from src.sampler import augment_sample, labels2output_map
# from src.utils import image_files_from_folder, show

from src.sampler_train import augment_sample, labels2output_map
from src.utils_train import image_files_from_folder, show
from src.loss import loss
from src.label import readShapes
from src.keras_utils import save_model, load_model
from os import makedirs
from os.path import isfile, isdir, basename, splitext
from random import choice
import keras
import argparse
import cv2
import numpy as np
import sys
import os
import tensorflow as tf

# from dataset2 import DataGenerator
from evaluator07 import eval3

from dataset4 import DataGenerator
# from evaluator_batch import eval2

from glob import glob
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

#     assert model_stride == 2 ** 4, \
#         'Make sure your model generates a feature map with resolution ' \
#         '16x smaller than the input'

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
    
    return data
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default='', type=str,
                        help='Path to previous model')
    parser.add_argument('-n', '--name', default='my-trained-model', type=str, help='Model name')
    parser.add_argument('-tr', '--train-dir', default='../utvm_data_new/train', type=str,
                        help='Input data directory for training')
    parser.add_argument('-va', '--val-dir', type=str, default="../utvm_data/val ",
                        help='Input data directory for validation')
    parser.add_argument('-its', '--iterations', type=int, default=30000,
                        help='Number of mini-batch iterations (default = 300.000)')
    parser.add_argument('-bs', '--batch-size', type=int, default=64, help='Mini-batch size (default = 32)')
    parser.add_argument('-od', '--output-dir', type=str, default='./', help='Output directory (default = ./)')
    parser.add_argument('-op', '--optimizer', type=str, default='Adam', help='Optmizer (default = Adam)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=.0001, help='Optmizer (default = 0.01)')
    args = parser.parse_args()

    netname = basename(args.name)
    train_dir = args.train_dir
    outdir = args.output_dir
    val_dir = args.val_dir

    iterations = args.iterations
    batch_size = args.batch_size
    
    print("\nbatch_size:", batch_size)
    print("args.model: ", args.model)
    print("output_dir:", outdir)
    print("train dir: ", train_dir)
    print("val dir: ", val_dir)
    
    model_path_backup = '%s/%s_backup' % (outdir, netname)
    model_path_backup_max = '%s/%s_backup_max' % (outdir, netname)
    model_path_final = '%s/%s_file' % (outdir, netname)
    model_path_final_max = '%s/%s_file_max' % (outdir, netname)
    
    best_ap =0.0
    best_ap_max = 0.0
    mean_loss = 0.0
    dim = 208

    iter_num = 50
    if not isdir(outdir):
        makedirs(outdir)
    
    

    f_max = open(outdir + "train_log_max.txt ", "a")
    f_max.write('start train\n')
    
    graph = tf.Graph()
    
    with  graph.as_default():
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session = tf.Session(config=session_config)
        with session.as_default():
            model, model_stride, xshape, yshape = load_network(args.model, dim)
            model.summary()
            opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
            model.compile(loss=loss, optimizer=opt)
             
#             train_data = read_data("/data/anhnt2/utvm_data_split/new_1008/bienso_train_T10_clear.txt")
#             test_data = read_data("/data/anhnt2/utvm_data_split/new_1008/bienso_test_T10_clear.txt")
            
            train_data = read_data("/data/anhnt2/utvm_data_split/new_1110/bienso_T10_2_train.txt")
            test_data = read_data("/data/anhnt2/utvm_data_split/new_1110/bienso_T10_2_test.txt")
            
#             train_data = read_data("/data/anhnt2/utvm_data_split/stanford/bienso_stanford_train.txt")
#             test_data = read_data("/data/anhnt2/utvm_data_split/stanford/bienso_stanford_test.txt")
#             test_data = read_data("/data/anhnt2/utvm_data_split/hard_case.txt")
            
            
            image_folder="/data/tagging_utvm/bienso/"

            training_generator = DataGenerator(train_data['img_dir'], train_data['annotation'],image_folder, dim=dim, model_stride=model_stride ,batch_size=8)
            test_generator = DataGenerator(test_data['img_dir'], test_data['annotation'],image_folder, dim=416,val=True,batch_size=32)
            
            
            print("eval:")
            #eval
#             h_list = []
#             w_list = []
            AP, R, P, AP_max, R_max, P_max,num_correct,num_correct_max, h_list, w_list  = eval3(model, test_generator)
            best_ap_max = AP_max
            
            #eval2
#             AP, R, P, AP_max, R_max, P_max,num_correct,num_correct_max  = eval2(model, test_generator)
#             best_ap_max = AP_max
            return
        
            #Ve distribution
            plt.scatter(w_list[1], h_list[1], s=3, color='red', label='Incorrect detection')
            plt.scatter(w_list[0], h_list[0], s=3, color='blue', label='Correct detection')
            plt.scatter(w_list[2], h_list[2], s=3, color='black', label='No detection')
            plt.legend()
            plt.title("Distribution of size of license plates")

            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.savefig('Distribution.png')
            # plt.show()
            return
        
if __name__ == '__main__':
    main()

''' 
eval:
python infer.py --model models/utvm_models_origin_2110/utvm-model-origin-2110_backup_max --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

python infer.py --model models/utvm_models_resnet18_2/utvm-model-resnet18-2_backup_max --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

python infer.py --model models/utvm_models_small_0311/utvm-model-small-0311_backup_max --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

python infer.py --model models/utvm_models_cus_small_mish/utvm-model-cus-small-mish_backup_max_backup_max --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

python infer.py --model models/utvm_models_small_0311/utvm-model-small-0311_backup_max --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

python infer.py --model models/model_goc/wpod-net_update1 --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

train:
python train-detector-utvm-backup.py --model models/origin_model/utvm-model_max_backup_max.h5 --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64

python train-detector-utvm-backup.py --name utvm-model-origin-noaug-pro01 --output-dir models/utvm_models_origin_noaug_pro01/ -op Adam -lr .0001 -its 100000 -bs 64

python infer.py --model models/utvm_models_cus20/utvm-model-cus20_backup_max --name utvm-model-eval --output-dir models/utvm_models_eval/ -op Adam -lr .0001 -its 100000 -bs 64


python infer.py --model models/utvm_models_resnet18_2/utvm-model-resnet18-2_backup_max  --name utvm-model-resnet18-2 --output-dir models/utvm_models_resnet18_2/ -op Adam -lr .0001 -its 100000 -bs 64

AP:  [0.58821996] R:  [0.945] P:  [0.60771704] total box: 400 correct:  378
AP_max:  [0.91458333] R_max:  [0.9425] P_max:  [0.96666667] total box: 400 correct:  377

'''

# moved all data to alpr-unconstrained-master-data
