import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import cv2
import traceback

from glob import glob
from src.utils import im2single
from src.keras_utils import decode_predict

import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
 
def eval2(model, eval_data_generator,iou_thres=0.5): 

    #     val_dir = '../ETC_data/val'
    
    out_dir = './output_train_densenet/' 
    lp_threshold = 0.0
    
    correct = []
    correct_max = []
    conf_list = []
    conf_list_max = []
    total_box = 0 
    count =0
    
    for i,data in enumerate(eval_data_generator):
        Xtrain, Ytrain,imgs_dir_list = data     
        img_dir = imgs_dir_list[0]
        file_name = img_dir.split("/")[-1].split("\\")[-1]
        folder_dir =img_dir.replace(file_name,"") 
        Ivehicle=Xtrain[0]
        Ivehicle = cv2.resize(Ivehicle,(208,208))
        real_labels = Ytrain[0]
        total_box +=len(real_labels)
#         print("i: ",i," real_labels:",len(real_labels))
        Iresized = im2single(Ivehicle)
        
        Iresized = Iresized.reshape((1,Iresized.shape[0],Iresized.shape[1],Iresized.shape[2]))
        Yr = model.predict(Iresized) 
        
        Yr= np.squeeze(Yr)
        predict_labels = decode_predict(Yr, Iresized[0], lp_threshold) 
        
        
        # write pred file
#         """
#         if (i+1) %(eval_data_generator.__len__()/100) ==0 :
        
        if (len(predict_labels)): 
            predict_labels_max = [predict_labels[0]] 
            
        else: 
#             if (i+1) %3 ==0 : 
            img_saved_path = out_dir +'no_prediction_'+file_name.split('.')[0]+'.png'
            cv2.imwrite(img_saved_path, Ivehicle)
            continue
        if len(real_labels) == 0: 
            continue
             
        this_conf_list_max, this_correct_max,img = get_correct_and_conf(predict_labels_max,  real_labels, iou_thres,Ivehicle,draw=True)
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
        
        
        
#         print("correct:")
#         print(np.sum(this_correct_max))
#         if os.path.exists(out_dir+folder_dir) == False:
#             os.makedirs(out_dir+folder_dir)
#         write_path = out_dir+folder_dir+file_name
#         img_saved_path = out_dir +folder_dir+file_name.split('.')[0]+'.png'
        
         
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
            # Extract index of largest overlap
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
#                 print("real_pts %d: "%(i),real_pts)
                [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = real_pts
                h,w = np.shape(Ivehicle)[:2]
                pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]               
#                 print("w,h:",(w,h))

#                 print("conf: ",conf)

                pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)

#                 print("real_pts:",pts_arr)

                Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(0,255,255),2)
            h,w = np.shape(Ivehicle)[:2]
            x0, x1, x2, x3, y0, y1, y2, y3 = np.reshape(label.pts,-1)
            pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]               
#             print("w,h:",(w,h))

#             print("conf: ",conf)

            pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)
#             print("pred_pts:",pred_pts)
#             print("pred_pts:",pts_arr)

            Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(255,0,0),2)
#             plt.imshow(Ivehicle)
#             plt.show() 
              
    return conf_list,correct,Ivehicle


def get_correct_and_conf2(predict_labels,real_labels,iou_thres,Ivehicle,max_stage = False):
    detected = []
    correct = []
    conf_list = []
    for label in predict_labels:
        
        x0, x1, x2, x3, y0, y1, y2, y3 = np.reshape(label.pts, -1)
         
        
        conf = label.prob()
        conf_list.append(conf)
        print()
        pred_pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        ###
        if max_stage ==True:
            
            h,w = np.shape(Ivehicle)[:2]
            pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]               
            print("w,h:",(w,h))

            print("conf: ",conf)

            pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)
            print("pred_pts:",pred_pts)
            print("pred_pts:",pts_arr)

            Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(0,0,255),2)

  
        ious = []
        for real_pts in real_labels:
            if max_stage ==True:
                print("real_pts:",real_pts)
                [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = real_pts
                h,w = np.shape(Ivehicle)[:2]
                pts2 = [(x0*w, y0*h), (x1*w, y1*h), (x2*w, y2*h), (x3*w, y3*h)]               
                print("w,h:",(w,h))

                print("conf: ",conf)

                pts_arr = np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)
                
                print("real_pts:",pts_arr)

                Ivehicle=cv2.polylines(Ivehicle,[pts_arr],True,(0,255,255),2)
            
             
            
            iou = iou_shapely(pred_pts, real_pts)
            ious.append(iou)
            # Extract index of largest overlap
            
        plt.imshow(Ivehicle)
        plt.show()  
        
        print()
        ious = np.array(ious)
        best_i = np.argmax(ious)

        # If overlap exceeds threshold and classification is correct mark as correct
        if ious[best_i] > iou_thres and best_i not in detected:
            correct.append(1)
            detected.append(best_i)
        else:
            correct.append(0)
    return conf_list,correct 

'''
correct:
51
AP:  [0.89279383]
R:  [0.98076923]
P:  [0.86440678]
end eval1 
'''



