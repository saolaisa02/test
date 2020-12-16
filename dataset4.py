import numpy as np
import keras
from glob import glob
import cv2
import os
from src.sampler import augment_sample, labels2output_map
# from src.sampler2 import augment_sample, labels2output_map
# from src.sampler_train import augment_sample, labels2output_map
# import matplotlib.pyplot as plt 
from src.utils import im2single
from src.keras_utils import decode_predict

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, imgs_dir, labels,image_folder,val=False, batch_size=16, dim=210, n_channels=3,
                 n_classes=10,   model_stride=16):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_dir = imgs_dir
        self.image_folder = image_folder
        self.model_stride = model_stride
        self.val = val
        self.n_channels = n_channels
        self.n_classes = n_classes 

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
        X, y,imgs_dir_list = self.__data_generation(indexes)
        return X, y,imgs_dir_list

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
            for i,idx in enumerate(indexes): 

                img_path = self.image_folder+self.imgs_dir[idx] 
                img_paths = glob(os.path.join(img_path))
#                 print("img_path:",img_path)
                img = cv2.imread(img_paths[0])
                # Store class 
                pts_label = self.readShapes2(self.labels[idx]) 
#                 plt.imshow(img)
#                 plt.show() 

                augment_img, pts_label,pts = self.process_data_item(img, pts_label[0], self.dim, self.model_stride)
                X[i,] = augment_img
    #             print("pts_label",np.shape(pts_label))
                y.append(pts_label)
                # show img
    #             plt.imshow(augment_img)
    #             plt.show() 
                # show img with polygon
                image_size = np.shape(augment_img)[0]
                pts2 = np.transpose(np.asarray(pts))

                pts2 = pts2.reshape((-1,1,2))*image_size
                pts2 = np.asarray(pts2 ,np.int32)  
#                 cv2.polylines(augment_img,[np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)],True,(0,255,255),2)
#                 plt.imshow(augment_img)
#                 plt.show() 
                # show optimal ground truth
#                 print("YY shape,",np.shape(pts_label))    
#                 plt.imshow(pts_label[:,:,0]*255 )
#                 plt.show()
                imgs_dir_list.append(self.imgs_dir[idx])

            return np.asarray(X), np.asarray(y),imgs_dir_list
        else:
            X  = []
            y = [] 
            imgs_dir_list = []
            for i,idx in enumerate(indexes): 

                img_path = self.image_folder+self.imgs_dir[idx] 
                img_paths = glob(os.path.join(img_path))
#                 print("img_path:",img_path)
                img = cv2.imread(img_paths[0])
                img = cv2.resize( im2single(img), (self.dim, self.dim))
                X.append(img)
                y.append(self.read_labelsf2(self.labels[idx]))
                imgs_dir_list.append(self.imgs_dir[idx])
                # Store class
#                 print(np.shape(X))
#                 print(np.shape(y))
            return  np.asarray(X),np.asarray(y),imgs_dir_list

             

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
        
#         image_size = np.shape(XX)[0]
        
#         pts2 = np.transpose(np.asarray(pts))

#         pts2 = pts2.reshape((-1,1,2))*image_size
#         pts2 = np.asarray(pts2 ,np.int32)  
#         cv2.polylines(XX,[np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)],True,(0,255,255),2)

        YY = labels2output_map(llp, pts, dim, model_stride)
         
#         print("YY shape,",np.shape(YY))    
#         plt.imshow(YY[:,:,0]*255 )
#         plt.show()
        
        
        return XX, YY,pts
    def read_labelsf2(self,lines):
        labels = [] 
        for line in lines:
            data 		= line.strip().split(',')
            ss 			= int(data[0])
            values 		= data[1:(ss*2 + 1)] 

            x0, x1, x2, x3, y0, y1, y2, y3= np.array([float(value) for value in values])              
            pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] 
            labels.append(pts)
        return labels

'''
imgs_dir = "../lp_data_new/cars_train/"
ann_dir = "../lp_data_new/cars_train/"

img_paths = glob(imgs_dir + "*.jpg")
ann_paths = glob(ann_dir + "*.txt")

training_generator = DataGenerator(imgs_dir, ann_paths, dim=208)
for epoch in range(5):
for data in training_generator:
        print("epoch: ", epoch)
        print(np.shape(data[0]))
        print(np.shape(data[1]))
  
        
''' 