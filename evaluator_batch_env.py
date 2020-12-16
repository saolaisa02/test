import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import cv2
import traceback

from glob import glob
# from src.utils import im2single
from src.keras_utils import decode_predict

import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
 
def eval2(model, eval_data_generator,iou_thres=0.65, env=""): 
    iou_thres=0.65
    lp_threshold = 0.1
     
    correct = []
    correct_max = []
    conf_list = []
    conf_list_max = []
    total_box = 0 
    count =0
    count_env = 0
    print(env)
    
    for i,data in enumerate(eval_data_generator):
        Xtrain, Ytrain,imgs_dir_list = data     
        
        img_dir = imgs_dir_list[0]
        if(env in img_dir):
            count_env += 1
            Yr_batch = model.predict(Xtrain) 

            for j,(Iresized,Yr,real_labels,img_dir) in enumerate(zip(Xtrain,Yr_batch,Ytrain,imgs_dir_list)): 

                file_name = img_dir.split("/")[-1].split("\\")[-1]
                folder_dir =img_dir.replace(file_name,"")  
                total_box +=len(real_labels) 
                predict_labels = decode_predict(Yr, Iresized, lp_threshold) 


                if (len(predict_labels)): 
                    predict_labels_max = [predict_labels[0]] 

                else:  
                    continue

                if len(real_labels) == 0: 
                    continue

                this_conf_list_max, this_correct_max,img = get_correct_and_conf(predict_labels_max,  real_labels, iou_thres,Iresized,draw=True)         
                correct_max +=this_correct_max 
                conf_list_max+=this_conf_list_max 
        else:
            continue
         
    print("Eval in ", env)
    AP_max, R_max, P_max = ap(tp=correct_max, conf=conf_list_max, n_gt = total_box)
    print("AP_max: ", AP_max,"R_max: ", R_max,"P_max: ", P_max,"Total box:",total_box, "Correct: ",np.sum(correct_max))
    print("\n\nTotal batch: ", count_env)
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
    
def read_labelsf(label_path):
    labels = []
    with open(label_path) as fp:
        for line in fp:
            data 		= line.strip().split(',')
            ss 			= int(data[0])
            values 		= data[1:(ss*2 + 1)] 
            
            x0, x1, x2, x3, y0, y1, y2, y3= np.array([float(value) for value in values])              
            pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] 
            labels.append(pts)
    return labels

def get_correct_and_conf(predict_labels,real_labels,iou_thres,Ivehicle,max_stage = False,draw=False):

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
            Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(255,0,0),2) 
              
    return conf_list,correct,Ivehicle

 
'''
correct:
51
AP:  [0.89279383]
R:  [0.98076923]
P:  [0.86440678]
end eval1 
'''



