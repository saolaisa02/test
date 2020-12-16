from pdb import set_trace as pause
from src.data_generator import DataGenerator
# from src.sampler import augment_sample, labels2output_map
# from src.utils import image_files_from_folder, show

from src.sampler import augment_sample, labels2output_map
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

from utvm_demo2 import eval

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

    assert model_stride == 2 ** 4, \
        'Make sure your model generates a feature map with resolution ' \
        '16x smaller than the input'

    return model, model_stride, input_shape, output_shape


def process_data_item(data_item, dim, model_stride):
#     print("data_item[1]:",data_item[0])
    if data_item[1] == None:
        stride = model_stride

        outsize = int(dim / stride)
        pts = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]).reshape((2, 4))
#         XX, llp, pts = augment_sample(data_item[0], pts, dim)
        
        XX = cv2.resize(data_item[0],(dim,dim))
        YY = np.zeros((outsize, outsize, 2 * 4 + 1), dtype='float32')
    else:
        XX, llp, pts = augment_sample(data_item[0], data_item[1].pts, dim)
        YY = labels2output_map(llp, pts, dim, model_stride)

    return XX, YY


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default='models/etc/model_v2_12600/my-trained-model_backupit_12600', type=str,
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
    best_ap =0.0
    best_ap_max = 0.0
    mean_loss = 0.0
    dim = 208
    iter_num = 30
#     if not isdir(outdir):
#         makedirs(outdir)

#     f = open(outdir + "train_log.txt ", "a")
#     f.write('start train\n')

#     f_max = open(outdir + "train_log_max.txt ", "a")
#     f_max.write('start train\n')
    graph = tf.Graph()
    with  graph.as_default():
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session = tf.Session(config=session_config)
        with session.as_default():
            #             self.wpod_net_path = wpod_net_path
            #             self.wpod_net = load_model(wpod_net_path)

            model, model_stride, xshape, yshape = load_network(args.model, dim)

#             opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
#             model.compile(loss=loss, optimizer=opt)

            eval(model) 

if __name__ == '__main__':
    main()

''' 

####
rescus
python eval-detector.py --model final_models/IoU_05/utvm_models_privatedata_resnet_cus/utvm-model-privatedata-resnet-cus_backup_max --name my-trained-model_max --train-dir ../utvm_data_new/train  --output-dir models/utvm_models/ -op Adam -lr .0001 -its 100000 -bs 64
###
eccv
python eval-detector.py --model final_models/IoU_05/utvm_models_eccv_relu_privatedata/utvm-model-eccv-relu-privatedata_backup_max --name my-trained-model_max --train-dir ../utvm_data_new/train  --output-dir models/utvm_models/ -op Adam -lr .0001 -its 100000 -bs 64
####
resnet2
python eval-detector.py --model final_models/IoU_05/utvm_models_resnet2/backup/utvm-model-privatedata-resnet2_backup_max-Copy1 --name my-trained-model_max --train-dir ../utvm_data_new/train  --output-dir models/utvm_models/ -op Adam -lr .0001 -its 100000 -bs 64

python eval-detector.py --model /models/utvm_models_origin_2110/utvm-model-origin-2110_backup_max --name my-trained-model_max --train-dir ../utvm_data_new/train  --output-dir models/utvm_models/ -op Adam -lr .0001 -its 100000 -bs 64

python eval-detector.py --model models/utvm_models/backup_utvm_new/my-trained-model_max_backup_max --name my-trained-model_max --train-dir ../utvm_data_new/train  --output-dir models/utvm_models/ -op Adam -lr .0001 -its 100000 -bs 64

python eval-detector.py --model models_official/utvm_models_resnet1_private_data/utvm-model-privatedata-resnet1_backup_max

AP:  [0.58821996] R:  [0.945] P:  [0.60771704] total box: 400 correct:  378
AP_max:  [0.91458333] R_max:  [0.9425] P_max:  [0.96666667] total box: 400 correct:  377

 
'''

# moved all data to alpr-unconstrained-master-data
