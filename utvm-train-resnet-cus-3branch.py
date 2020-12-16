from src.keras_utils import save_model
from os import makedirs
from os.path import isfile, isdir, basename, splitext 
import keras
import argparse
import cv2
import numpy as np 
import os
import tensorflow as tf 
from glob import glob
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow 
from src.activation_function import *
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input, Dropout
from keras.models import Model

from src.keras_utils import decode_predict
from shapely import geometry
####

import random
from src.utils import im2single, getWH, hsv_transform, IOU_centre_and_dims
from src.label import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def l1(true,pred,szs):
    b,h,w,ch = szs
    res = tf.reshape(true-pred,(b,h*w*ch))
    res = tf.abs(res)
    res = tf.reduce_sum(res,1)
    return res
 
def logloss(Ptrue,Pred,szs,eps=10e-10):
    alpha = 2.
    
    b,h,w,ch = szs
    Pred = tf.clip_by_value(Pred,eps,1.)
    Pred = -tf.log(Pred)
    
    Pred = Pred*Ptrue
    Pred = tf.reshape(Pred,(b,h*w*ch))
    Pred = tf.reduce_sum(Pred,1)
    return Pred


def loss(Ytrue, Ypred):

    b = tf.shape(Ytrue)[0]
    h = tf.shape(Ytrue)[1]
    w = tf.shape(Ytrue)[2]

    obj_probs_true = Ytrue[...,0]
    obj_probs_pred = Ypred[...,0]

    non_obj_probs_true = 1. - Ytrue[...,0]
    non_obj_probs_pred = Ypred[...,1]

    affine_pred	= Ypred[...,2:]
    pts_true 	= Ytrue[...,1:]

    affinex = tf.stack([tf.maximum(affine_pred[...,0],0.),affine_pred[...,1],affine_pred[...,2]],3)
    affiney = tf.stack([affine_pred[...,3],tf.maximum(affine_pred[...,4],0.),affine_pred[...,5]],3)

    v = 0.5
    base = tf.stack([[[[-v,-v,1., v,-v,1., v,v,1., -v,v,1.]]]])
    base = tf.tile(base,tf.stack([b,h,w,1]))

    pts = tf.zeros((b,h,w,0))

    for i in range(0,12,3):
        row = base[...,i:(i+3)]
        ptsx = tf.reduce_sum(affinex*row,3)
        ptsy = tf.reduce_sum(affiney*row,3)

        pts_xy = tf.stack([ptsx,ptsy],3)
        pts = (tf.concat([pts,pts_xy],3))

    flags = tf.reshape(obj_probs_true,(b,h,w,1))
    res   = 1.*l1(pts_true*flags,pts*flags,(b,h,w,4*2))
    
    res  += 1.*logloss(obj_probs_true,obj_probs_pred,(b,h,w,1))
    res  += 1.*logloss(non_obj_probs_true,non_obj_probs_pred,(b,h,w,1))
    
    return res

def iou_shapely(pts_1, pts_2):
    polygon_1 = geometry.Polygon([(p[0], p[1]) for p in pts_1])
    polygon_2 = geometry.Polygon([(p[0], p[1]) for p in pts_2])
    area_1 = polygon_1.area
    area_2 = polygon_2.area
    intersection_area = polygon_1.intersection(polygon_2).area
    union_area = area_1 + area_2 - intersection_area
    return intersection_area/union_area

def write_labelsf(fp,label_path,pts,text):
    fp.write('%d,' % 4)
    ptsarray = pts.flatten()
    fp.write(''.join([('%f,' % value) for value in ptsarray]))
    fp.write('%s,' % text)
    fp.write('\n') 

def get_correct_and_conf(predict_labels,real_labels,iou_thres,img,max_stage = False,draw=False):
    Ivehicle = img*255

    detected = []
    correct = []
    conf_list = []
    for label in predict_labels: 
        x0, x1, x2, x3, y0, y1, y2, y3 = np.reshape(label.pts,-1)
        conf = label.prob()
        conf_list.append(conf) 
        pred_pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] 

        ious = []
        for real_pts in real_labels:
            iou = iou_shapely(pred_pts, real_pts)
            ious.append(iou) 
        ious = np.array(ious)
        best_i = np.argmax(ious)

        # If overlap exceeds threshold and classification is correct mark as correct
        if ious[best_i] > iou_thres and best_i not in detected:
            correct.append(1)
            detected.append(best_i)
        else:
            correct.append(0)
        if draw==True:

            for i,real_pts in enumerate(real_labels):  
                [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = real_pts
                h,w = np.shape(Ivehicle)[:2]
                pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]    

                pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32) 

                Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(0,255,255),2)
            h,w = np.shape(Ivehicle)[:2]
            x0, x1, x2, x3, y0, y1, y2, y3 = np.reshape(label.pts,-1)
            pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]     

            pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32) 
#             Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(255,0,0),2) 

    return conf_list,correct,Ivehicle

def ap(tp, conf, n_gt):

    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        n_sample_per_cls: Number samples per class (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf = np.array(tp), np.array(conf)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []

    n_p = sum(i)  # Number of predicted objects

    if (n_p == 0) or (n_gt == 0):
        ap.append(0)
        r.append(0)
        p.append(0)
    else:
        # Accumulate FPs and TPs
        fpc = np.cumsum(1 - tp[i])
        tpc = np.cumsum(tp[i])
        print("true positive: ",tpc[-1],"false positive: ",fpc[-1],"n_gt: ",n_gt," total positive results:",(tpc[-1] + fpc[-1]))    
        # Recall
        recall_curve = tpc*1.0 / (n_gt + 1e-16)
        r.append(tpc[-1]*1.0 / (n_gt + 1e-16))

        # Precision
        precision_curve = tpc*1.0 / (tpc + fpc)
        p.append(tpc[-1]*1.0 / (tpc[-1] + fpc[-1]))

        # AP from recall-precision curve
        ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap 
def eval2(model, eval_data_generator,iou_thres=0.5): 

    #     val_dir = '../ETC_data/val'
    
    out_dir = './output_new28/' 
    iou_thres=0.5
    lp_threshold = 0.1
     
    correct = []
    correct_max = []
    conf_list = []
    conf_list_max = []
    total_box = 0 
    count =0
    
    for i,data in enumerate(eval_data_generator):
        Xtrain, Ytrain,imgs_dir_list = data     
         

        Yr_batch = model.predict(Xtrain) 
         
        for j,(Iresized,Yr,real_labels,img_dir) in enumerate(zip(Xtrain,Yr_batch,Ytrain,imgs_dir_list)): 
            
            file_name = img_dir.split("/")[-1].split("\\")[-1]
            folder_dir =img_dir.replace(file_name,"")  
            total_box +=len(real_labels) 
            predict_labels = decode_predict(Yr, Iresized, lp_threshold) 
         
        
            if (len(predict_labels)): 
                predict_labels_max = [predict_labels[0]] 

            else:  
                img_saved_path = out_dir +'no_prediction_'+file_name.split('.')[0]+'.png'
                cv2.imwrite(img_saved_path, Iresized)
                continue
                
            if len(real_labels) == 0: 
                continue
             
            this_conf_list_max, this_correct_max,img = get_correct_and_conf(predict_labels_max,  real_labels, iou_thres,Iresized,draw=True)
            
            """
            if (i+1) %1 ==0 :   
                if os.path.exists(out_dir) == False:
                    os.makedirs(out_dir)
                if sum(this_correct_max) ==0:
                    img_saved_path = out_dir +'incorrect_'+file_name.split('.')[0]+'.png'
                    cv2.imwrite(img_saved_path, img)
                    print("img_saved_path: ",img_saved_path)
                else:
                    img_saved_path = out_dir +'correct_'+file_name.split('.')[0]+'.png'
                    print("img_saved_path: ",img_saved_path) 
                    cv2.imwrite(img_saved_path, img)

                write_path = out_dir+file_name.split('.')[0]+".txt"
    #             with open(write_path,'w') as fp: 

    #                 for label in predict_labels_max:
    #                     x0, x1, x2, x3, y0, y1, y2, y3 =  np.reshape(label.pts,-1)
    #                     pred_pts = np.array([x0, x1, x2, x3, y0, y1, y2, y3] )
    #                     write_labelsf(fp,write_path,pred_pts,'PLATE')


    #         """
        
        
        
#             print("correct:")
#             print(np.sum(this_correct_max))
#             if os.path.exists(out_dir+folder_dir) == False:
#                 os.makedirs(out_dir+folder_dir)
#             write_path = out_dir+folder_dir+file_name
#             img_saved_path = out_dir +folder_dir+file_name.split('.')[0]+'.png'
        
         
            correct_max +=this_correct_max 
            conf_list_max+=this_conf_list_max 
         
    print("maximum score method:")
    AP_max, R_max, P_max = ap(tp=correct_max, conf=conf_list_max, n_gt = total_box)
    # Compute mean AP across all classes in this image, and append to image list
    print()  
    print("AP_max: ", AP_max,"R_max: ", R_max,"P_max: ", P_max,"total box:",total_box, "correct: ",np.sum(correct_max))
    return 0, 0,0,AP_max[0], R_max[0], P_max[0],0,np.sum(correct_max)


def ap(tp, conf, n_gt):

    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        n_sample_per_cls: Number samples per class (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf = np.array(tp), np.array(conf)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []

    n_p = sum(i)  # Number of predicted objects

    if (n_p == 0) or (n_gt == 0):
        ap.append(0)
        r.append(0)
        p.append(0)
    else:
        # Accumulate FPs and TPs
        fpc = np.cumsum(1 - tp[i])
        tpc = np.cumsum(tp[i])
        print("true positive: ",tpc[-1],"false positive: ",fpc[-1],"n_gt: ",n_gt," total positive results:",(tpc[-1] + fpc[-1]))    
        # Recall
        recall_curve = tpc*1.0 / (n_gt + 1e-16)
        r.append(tpc[-1]*1.0 / (n_gt + 1e-16))

        # Precision
        precision_curve = tpc*1.0 / (tpc + fpc)
        p.append(tpc[-1]*1.0 / (tpc[-1] + fpc[-1]))

        # AP from recall-precision curve
        ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap 

def labels2output_map(label,lppts,dim,stride): 
    side = ((float(dim) + 40.)/2.)/stride # 7.75 when dim = 208 and stride = 16

    outsize = int(dim/stride)
    Y  = np.zeros((outsize,outsize,2*4+1),dtype='float32')
    MN = np.array([outsize,outsize])
    WH = np.array([dim,dim],dtype=float)

    tlx,tly = np.floor(np.maximum(label.tl(),0.)*MN).astype(int).tolist()
    brx,bry = np.ceil (np.minimum(label.br(),1.)*MN).astype(int).tolist()
    max_iou = 0.0
    x_max = -1
    y_max= -1
    for x in range(tlx,brx):
        for y in range(tly,bry):

            mn = np.array([float(x) + .5, float(y) + .5]) 
            iou = IOU_centre_and_dims(mn/MN,label.wh(),label.cc(),label.wh()) 
            if max_iou<iou:
                max_iou = iou
                x_max = x
                y_max = y
                p_WH = lppts*WH.reshape((2,1))
                p_MN = p_WH/stride 
                p_MN_center_mn = p_MN - mn.reshape((2,1)) 
                p_side_max = p_MN_center_mn/side
            if iou> .6 :
                p_WH = lppts*WH.reshape((2,1))
                p_MN = p_WH/stride

                p_MN_center_mn = p_MN - mn.reshape((2,1))

                p_side = p_MN_center_mn/side

                Y[y,x,0] = 1.


                Y[y,x,1:] = p_side.T.flatten()
    if x_max>0 and y_max >0: 
        # trong feature map it nhat co 1 cell duoc activate, do la cell co iou lon nhat
        Y[y_max,x_max,0] = 1. 
        Y[y_max,x_max,1:] = p_side_max.T.flatten()

    return Y


def pts2ptsh(pts):
    return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))


def project(I, T, pts, dim):
    ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))
    ptsh = np.matmul(T, ptsh)
    ptsh = ptsh / ptsh[2]
    ptsret = ptsh[:2]
    ptsret = ptsret / dim
    Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=.0, flags=cv2.INTER_LINEAR)
    return Iroi, ptsret


def flip_image_and_pts(I, pts):
    I = cv2.flip(I, 1)
    pts[0] = 1. - pts[0]
    idx = [1, 0, 3, 2]
    pts = pts[..., idx]
    return I, pts

def augment_sample(I, pts, dim):
    maxsum, maxangle = 120, np.array([30., 30., 20.])
    angles = np.random.rand(3) * maxangle
    if angles.sum() > maxsum:
        angles = (angles / angles.sum()) * (maxangle / maxangle.sum())

    I = im2single(I)


    prob = random.random() 

    iwh = getWH(I.shape) 
    pts = pts * iwh.reshape((2, 1))
    
    wsiz = (max (pts[0,:])-min(pts[0,:]))
    hsiz = (max (pts[1,:])-min(pts[1,:]))
    dx = random.uniform(0., dim - wsiz)
    dy = random.uniform(0., dim - hsiz)
    deltax = pts[0,0]-dx
    deltay = pts[1,0]-dy
    pph=np.matrix([
        [pts[0,0]-deltax,pts[0,1]-deltax,pts[0,2]-deltax,pts[0,3]-deltax],
        [pts[1,0]-deltay,pts[1,1]-deltay,pts[1,2]-deltay,pts[1,3]-deltay],
        [1.,1.,1.,1.]],
        dtype=float)
    T = find_T_matrix(pts2ptsh(pts), pph)

    H = perspective_transform((dim, dim), angles=np.array([0., 0., 0.]))
    H = np.matmul(H, T) 

    Iroi, pts = project(I, H, pts, dim)

#     hsv_mod = np.random.rand(3).astype('float32')
#     hsv_mod = (hsv_mod - .5) * .3
#     hsv_mod[0] *= 360
#     Iroi = hsv_transform(Iroi, hsv_mod)
#     Iroi = np.clip(Iroi, 0., 1.)

    pts = np.array(pts)
 

    tl, br = pts.min(1), pts.max(1)
    llp = Label(0, tl, br) 
    return Iroi, llp, pts


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, imgs_dir, labels, image_folder, val=False, batch_size=32, dim=208, n_channels=3, model_stride=16):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_dir = imgs_dir
        self.image_folder = image_folder
        self.model_stride = model_stride
        self.val = val
        self.n_channels = n_channels
        self.dim = dim

        # self.on_epoch_start()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if index == 0:
            self.on_epoch_start()
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y, imgs_dir_list = self.__data_generation(indexes)
        return X, y, imgs_dir_list

    def on_epoch_start(self):
        'Updates indexes before each epoch'

        self.indexes = np.arange(len(self.labels))

        if self.val == False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        if self.val == False:
            X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels))
            y = []
            imgs_dir_list = []
            for i, idx in enumerate(indexes):
                img_path = self.image_folder + self.imgs_dir[idx]
                img_paths = glob(os.path.join(img_path))
                #                 print("img_path:",img_path)
                img = cv2.imread(img_paths[0])
                # Store class
                pts_label = self.readShapes2(self.labels[idx])
                #                 plt.imshow(img)
                #                 plt.show()

                augment_img, pts_label, pts = self.process_data_item(img, pts_label[0], self.dim, self.model_stride)
                X[i,] = augment_img
                #             print("pts_label",np.shape(pts_label))
                y.append(pts_label)
                # show img
                #             plt.imshow(augment_img)
                #             plt.show()
                # show img with polygon
                image_size = np.shape(augment_img)[0]
                pts2 = np.transpose(np.asarray(pts))

                pts2 = pts2.reshape((-1, 1, 2)) * image_size
                pts2 = np.asarray(pts2, np.int32)
                #                 cv2.polylines(augment_img,[np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)],True,(0,255,255),2)
                #                 plt.imshow(augment_img)
                #                 plt.show()
                # show optimal ground truth
                #                 print("YY shape,",np.shape(pts_label))
                #                 plt.imshow(pts_label[:,:,0]*255 )
                #                 plt.show()
                imgs_dir_list.append(self.imgs_dir[idx])

            return np.asarray(X), np.asarray(y), imgs_dir_list
        else:
            X = []
            y = []
            imgs_dir_list = []
            for i, idx in enumerate(indexes):
                img_path = self.image_folder + self.imgs_dir[idx]
                img_paths = glob(os.path.join(img_path))
                #                 print("img_path:",img_path)
                img = cv2.imread(img_paths[0])
                img = cv2.resize(im2single(img), (self.dim, self.dim))
                X.append(img)
                y.append(self.read_labelsf2(self.labels[idx]))
                imgs_dir_list.append(self.imgs_dir[idx])
                # Store class
            return np.asarray(X), np.asarray(y), imgs_dir_list

    def readShapes(self, path):
        shapes = []
        with open(path) as fp:
            for line in fp:
                pts = self.read_label(line)
                shapes.append(pts)
                break
        return shapes

    def readShapes2(self, lines):
        shapes = []
        #         print(lines)
        for line in lines:
            pts = self.read_label(line)
            shapes.append(pts)
            break
        return shapes

    def read_label(self, line):
        data = line.strip().split(',')
        ss = int(data[0])
        values = data[1:(ss * 2 + 1)]
        text = data[(ss * 2 + 1)] if len(data) >= (ss * 2 + 2) else ''
        pts = np.array([float(value) for value in values]).reshape((2, ss))
        text = text
        return pts

    def process_data_item(self, img, pts_label, dim, model_stride):
        XX, llp, pts = augment_sample(img, pts_label, dim)
        YY = labels2output_map(llp, pts, dim, model_stride)
        return XX, YY, pts

    def read_labelsf2(self, lines):
        labels = []
        for line in lines:
            data = line.strip().split(',')
            ss = int(data[0])
            values = data[1:(ss * 2 + 1)]

            x0, x1, x2, x3, y0, y1, y2, y3 = np.array([float(value) for value in values])
            pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
            labels.append(pts)
        return labels


def res_block_mish(x,sz,filter_sz=3, down=False, up_size=False, pooling=False):
    if(up_size==True):
        x = Conv2D(sz, 1, activation='linear', padding='same', strides=(1,1), bias=False)(x)
        x = BatchNormalization()(x)
    if(pooling==True):
        xi = x
    else:
        xi  = Mish()(x)
    if(down==False):
        x_linear = x #get linear
        xi  = Conv2D(sz, filter_sz, activation='linear', padding='same', strides=(1,1))(xi)
        xi  = BatchNormalization()(xi)
        xi  = Mish()(xi)
    else:
        x_linear = MaxPooling2D(pool_size=(2,2))(x)
        xi  = Conv2D(sz, filter_sz, activation='linear', padding='same', strides=(2,2))(xi)
        xi  = BatchNormalization()(xi)
        xi  = Mish()(xi)

    xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
    xi  = BatchNormalization()(xi)
    
    xi  = Add()([xi,x_linear]) #concate
    return xi

def end_block(x):
    xprobs = Conv2D(2, 3, activation='softmax', padding='same')(x)
    xbbox = Conv2D(6, 3, activation='linear', padding='same')(x)
    return Concatenate(3)([xprobs, xbbox])
    
def create_model_resnet_cus():
    print("Model cus resnet cus 2 branchhhhhhhhhhhhhhhhhhhhhhhh\n--------------------------------------------------------")
    input_layer = Input(shape=(None,None,3),name='input')
    
    x = Conv2D(32, 5, activation='linear', padding='same', strides=(2,2))(input_layer)
    x = BatchNormalization()(x)
#     x = Mish()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x_branch1 = Conv2D(32, 3, activation='linear', padding='same', strides=(1,1))(x)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,32, pooling=True)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
    
    x_branch1 = Conv2D(32, 3, activation='linear', padding='same', strides=(1,1))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,32)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
        
    x_branch1 = Conv2D(64, 3, activation='linear', padding='same', strides=(2,2))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,64, down=True, up_size=True)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
    x_branch2 = Mish()(x)
    
    x_branch2 = Conv2D(64, (5,1), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = Conv2D(64, (1,5), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = BatchNormalization()(x_branch2)
    x_branch1 = Conv2D(64, 3, activation='linear', padding='same', strides=(1,1))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,64)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
    x_branch2  = Add()([x, x_branch2])
    x_branch2 = Mish()(x_branch2)
     
    x_branch2 = Conv2D(128, (5,1), activation='linear', padding='same', strides=(2,2))(x_branch2)
    x_branch2 = Conv2D(128, (1,5), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = BatchNormalization()(x_branch2)
    x_branch1 = Conv2D(128, 3, activation='linear', padding='same', strides=(2,2))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,128, down=True, up_size=True)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
    x_branch2  = Add()([x, x_branch2])
    x_branch2 = Mish()(x_branch2)
    x_branch3 = Mish()(x)
    
    x_branch3 = Conv2D(128, (7,1), activation='linear', padding='same', strides=(1,1))(x_branch3)
    x_branch3 = Conv2D(128, (1,7), activation='linear', padding='same', strides=(1,1))(x_branch3)
    x_branch3 = BatchNormalization()(x_branch3)
    x_branch2 = Conv2D(128, (5,1), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = Conv2D(128, (1,5), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = BatchNormalization()(x_branch2)
    x_branch1 = Conv2D(128, 3, activation='linear', padding='same', strides=(1,1))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,128)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
    x_branch2  = Add()([x, x_branch2])
    x_branch2 = Mish()(x_branch2)
    x_branch3  = Add()([x, x_branch3])
    x_branch3 = Mish()(x_branch3)
    
    x_branch3 = Conv2D(256, (7,1), activation='linear', padding='same', strides=(1,1))(x_branch3)
    x_branch3 = Conv2D(256, (1,7), activation='linear', padding='same', strides=(1,1))(x_branch3)
    x_branch3 = BatchNormalization()(x_branch3)
    x_branch2 = Conv2D(256, (5,1), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = Conv2D(256, (1,5), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = BatchNormalization()(x_branch2)
    x_branch1 = Conv2D(256, 3, activation='linear', padding='same', strides=(1,1))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,256, up_size=True)
    x_branch1  = Add()([x, x_branch1])
    x_branch1 = Mish()(x_branch1)
    x_branch2  = Add()([x, x_branch2])
    x_branch2 = Mish()(x_branch2)
    x_branch3  = Add()([x, x_branch3])
    x_branch3 = Mish()(x_branch3)
    
    x_branch3 = Conv2D(256, (7,1), activation='linear', padding='same', strides=(1,1))(x_branch3)
    x_branch3 = Conv2D(256, (1,7), activation='linear', padding='same', strides=(1,1))(x_branch3)
    x_branch3 = BatchNormalization()(x_branch3)
    x_branch2 = Conv2D(256, (5,1), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = Conv2D(256, (1,5), activation='linear', padding='same', strides=(1,1))(x_branch2)
    x_branch2 = BatchNormalization()(x_branch2)
    x_branch1 = Conv2D(256, 3, activation='linear', padding='same', strides=(1,1))(x_branch1)
    x_branch1 = BatchNormalization()(x_branch1)
    x = res_block_mish(x,256)
    x = Add()([x, x_branch1, x_branch2, x_branch3])
#     x_branch1 = Mish()(x_branch1)
    x = Mish()(x)
    
    
    x = end_block(x)

    return Model(inputs=input_layer,outputs=x)

def load_model(path, custom_objects={}, verbose=0): 

    path = splitext(path)[0]
    model = create_model_resnet_cus()
#     model.summary()
    #     model = create_model_mobnet()
    #     model = create_model_DenseNet121()
    if path != '':
        model.load_weights('%s.h5' % path)
    if verbose: print('Loaded from %s' % path)
    return model


def load_network(modelpath, input_dim=208):
    model = load_model(modelpath)
    input_shape = (input_dim, input_dim, 3)

    # Fixed input size for training
    inputs = keras.layers.Input(shape=(input_dim, input_dim, 3))
    outputs = model(inputs)

    output_shape = tuple([s.value for s in outputs.shape[1:]])
    output_dim = output_shape[1]
    model_stride = input_dim / output_dim

    print("model_stride:", model_stride)
    print("input_dim: ", input_dim)
    print("output_dim: ", output_dim)

    assert input_dim % output_dim == 0, \
        'The output resolution must be divisible by the input resolution'

    #     assert model_stride == 2 ** 4, \
    #         'Make sure your model generates a feature map with resolution ' \
    #         '16x smaller than the input'

    return model, model_stride, input_shape, output_shape


#####
   
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
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='Mini-batch size (default = 32)')
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
    
#     print("\nbatch_size:", batch_size)
#     print("args.model: ", args.model)
#     print("output_dir:", outdir)
#     print("train dir: ", train_dir)
#     print("val dir: ", val_dir)
#     print("Learning rate: ", args.learning_rate)
    
    model_path_backup = '%s/%s_backup' % (outdir, netname)
    model_path_backup_max = '%s/%s_backup_max' % (outdir, netname)
    model_path_final = '%s/%s_file' % (outdir, netname)
    model_path_final_max = '%s/%s_file_max' % (outdir, netname)
    if not isdir(outdir):
        makedirs(outdir)
    best_ap =0.0
    best_ap_max = 0.0
    f_max = open(outdir + "/train_log_max.txt ", "a")
    f_max.write('start train\n')
    mean_loss = 0.0
    dim = 208
    iter_num = 50
#     AP_max = 0

    
    graph = tf.Graph()
    
    with  graph.as_default():
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session = tf.Session(config=session_config)
        with session.as_default():
            model, model_stride, xshape, yshape = load_network(args.model, dim)
            
            ####
            model.summary()
            print("\nbatch_size:", batch_size)
            print("args.model: ", args.model)
            print("output_dir:", outdir)
            print("train dir: ", train_dir)
            print("val dir: ", val_dir)
            print("Learning rate: ", args.learning_rate)
#             return 
            ####
            
            opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
            model.compile(loss=loss, optimizer=opt)

            train_data = read_data("/data/anhnt2/utvm_data_split/new_1008/bienso_train_T10_clear.txt")
            test_data = read_data("/data/anhnt2/utvm_data_split/new_1008/bienso_test_T10_clear.txt")

#             train_data = read_data("/data/anhnt2/utvm_data_split/stanford/bienso_stanford_train.txt")
#             test_data = read_data("/data/anhnt2/utvm_data_split/stanford/bienso_stanford_test.txt")
            
            image_folder="/data/tagging_utvm/bienso/"

            training_generator = DataGenerator(
                train_data['img_dir'], 
                train_data['annotation'],
                image_folder, 
                dim=208,
                batch_size=batch_size)

            test_generator = DataGenerator(
                test_data['img_dir'], 
                test_data['annotation'],
                image_folder, 
                dim=416,
                val=True,
                batch_size=32)
            
            h_list = []
            w_list = []

            AP, R, P, AP_max, R_max, P_max, num_correct, num_correct_max = eval2(model, test_generator)
            best_ap_max = AP_max 

            it = 0
            # return
            it_num = 0 
            mean_loss=0
            print("training_generator.__len__():",training_generator.__len__())
            for epoch in range(100000):
                print("epoch:", epoch, " batch size:", batch_size)
                
                for data in training_generator:
                    Xtrain, Ytrain, imgs_dir_list = data

                    train_loss = model.train_on_batch(Xtrain, Ytrain) 
                    mean_loss += train_loss 
                    it += 1 
                    it_num+=1
                    if (it + 1) % (training_generator.__len__() / 100) == 0:
                        print('\nIter. %d (of %d)' % (it + 1, iterations))
                        print('\tLoss: %f' % (mean_loss / it_num))
                    
                    
                    

                if epoch % 3 == 0:
                    print('\nIter. %d (of %d)' % (it + 1, iterations))
                    print('\tLoss: %f' % (mean_loss / it))
                    AP, R, P, AP_max, R_max, P_max, num_correct, num_correct_max = eval2(model, test_generator)
                    print("AP_max")
                    print(best_ap_max)
                    print(AP_max)
                    if best_ap_max < AP_max:
                        best_ap_max = AP_max
                        save_model(model, model_path_backup_max)
                        #                         save_model(model ,model_path_backup+'_it_%d'%(it+1))
                        f_max.write('iter: %d, best ap_max: %f ,n_correct: %d, Loss: %f\n' % (
                        it + 1, AP_max, num_correct_max, mean_loss / it_num))
                        f_max.flush()
                    else:
                        f_max.write('iter: %d,ap_max: %f, n_correct: %d, Loss: %f, ap: %f\n' % (
                        it + 1, AP_max, num_correct_max, mean_loss / it_num, AP))
                        f_max.flush() 
                    it_num = 0 
                    mean_loss=0

    print('Stopping data generator')
    f.close()
    f_max.close()

    print('Saving model (%s)' % model_path_final)
    save_model(model, model_path_final)


if __name__ == '__main__':
    main()
''' 
python utvm-train-resnet-cus-3branch.py --name utvm-model-privatedata-resnet-cus-3branch --output-dir models_official/utvm_models_privatedata_resnet_cus_3branch -op Adam -lr .0001 -its 100000 -bs 32
###

--model models_official/utvm_models_resnet_cus2/backup/utvm-model-stanford-resnet-cus2_backup_max

python utvm-train-resnet-cus2.py  --model models_official/utvm_models_resnet_cus2/backup/utvm-model-stanford-resnet-cus2_backup_max  --name utvm-model-stanford-resnet-cus2 --output-dir models_official/utvm_models_resnet_cus2 -op Adam -lr .001 -its 100000 -bs 32

python utvm-train-resnet-cus2.py   --name utvm-model-stanford-resnet-cus2 --output-dir models_official/utvm_models_resnet_cus2 -op Adam -lr .001 -its 100000 -bs 32

python utvm-train-resnet2.py --model models_official/utvm_models_resnet2_private_data/utvm-model-private-data-resnet2_backup_max --name utvm-model-stanford-resnet2 --output-dir models_official/utvm_models_resnet2 -op Adam -lr .001 -its 100000 -bs 32


python utvm-train-resnet-cus-3branch.py --model models_official/utvm_models_resnet_cus_3branch/utvm-model-stanford-resnet-cus-3branch_backup_max --name utvm-model-stanford-resnet-cus-3branch --output-dir models_official/utvm_models_resnet_cus_3branch -op Adam -lr .0001 -its 100000 -bs 32
'''