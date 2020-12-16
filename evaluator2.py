import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import cv2
import traceback

from glob import glob
from src.utils import im2single
from src.keras_utils import decode_predict

import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt

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

def read_labels(label_path):
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

def eval(model, iou_thres=0.8):
    val_dir = '../license-plate-detector-data/val'
    mean_mAP, mean_R, mean_P = 0.0, 0.0, 0.0
    mAPs, mR, mP = [], [], []
    lp_threshold = 0.6
    imgs_paths = glob('%s/Viet_VVT_POC_image_20200204/*/*.jpg' % val_dir) 
    # 		imgs_paths = glob('%s/*.png' % input_dir)
    print("number of val images:", len(imgs_paths))
    correct = []
    conf_list = []
    total_box = 0
    for i, img_path in enumerate(imgs_paths):
#         if i == 10:break
        labfile = img_path.replace('.jpg', '.txt')
        if os.path.exists(labfile)==False:
            continue
        real_labels = read_labels(labfile)
        total_box +=len(real_labels)
        Ivehicle = cv2.imread(img_path)
         
        Iresized = im2single(Ivehicle)
 

        Iresized = Iresized.reshape((1,Iresized.shape[0],Iresized.shape[1],Iresized.shape[2]))

        Yr 		= model.predict(Iresized)
        Yr 		= np.squeeze(Yr)

        predict_labels = decode_predict(Yr, Iresized[0], lp_threshold)
        
#         if (len(predict_labels)>0):
#             predict_labels = [predict_labels[0]]
#             print("total prediction:",len(predict_labels))
#             predict_labels = predict_labels[0:10]
#         else:
#             continue

        if len(predict_labels) == 0:
            # If there are no detections but there are labels mask as zero AP
            if len(real_labels) != 0:
                mAPs.append(0), mR.append(0), mP.append(0)
            continue

        # If no labels add number of detections as incorrect
        if len(real_labels) == 0:
            # correct.extend([0 for _ in range(len(detections))])
            mAPs.append(0), mR.append(0), mP.append(0)
            continue
        else:
            detected = []
          
            for label in predict_labels:  
                x0, x1, x2, x3, y0, y1, y2, y3 = np.reshape(label.pts,-1)  
                conf = label.prob()
                conf_list.append(conf)
                pred_pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
                h,w = np.shape(Ivehicle)[:2]
                pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]               
#                 print("w,h:",(w,h))
                
#                 print("conf: ",conf)
                
#                 pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)
#                 print("pred_pts:",pts_arr)
                
#                 Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(0,255,255),2)
                
                
#                 plt.imshow(Ivehicle)
#                 plt.show()
     
                
                ious = []
                for real_pts in real_labels:
#                     print("ground truth: ",real_pts)
#                     print("pred_pts: ",pred_pts)
                    iou = iou_shapely(pred_pts, real_pts)
                    ious.append(iou)
#                     print("iou:",iou)
                # Extract index of largest overlap
                ious = np.array(ious)
                best_i = np.argmax(ious)
                
                # If overlap exceeds threshold and classification is correct mark as correct
                if ious[best_i] > iou_thres and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

        # Compute Average Precision (AP) per class
       
    AP, R, P = ap(tp=correct, conf=conf_list, n_gt = total_box) 
    # Compute mean AP across all classes in this image, and append to image list
    
    print("total box:",total_box)
    print("correct:")
    
    print(np.sum(correct))
    
    print("AP: ", AP)
    print("R: ", R)
    print("P: ", P)
    
#     print("mAP: ", mean_mAP)
#     print("mean R: ", mean_R)
#     print("mean P: ", mean_P)

    return AP[0], R[0], P[0]


'''
correct:
51
AP:  [0.89279383]
R:  [0.98076923]
P:  [0.86440678]
end eval1 
'''
