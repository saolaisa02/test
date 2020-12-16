import numpy as np
import keras
from glob import glob
import cv2
import os
from src.sampler import augment_sample, labels2output_map
import matplotlib.pyplot as plt


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, imgs_dir, labels, batch_size=32, dim=208, n_channels=3,
                 n_classes=10, shuffle=True, model_stride=16):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.imgs_dir = imgs_dir
        self.model_stride = model_stride

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

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

        # Find list of IDs
        list_labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_labels_temp)
        return X, y

    def on_epoch_start(self):
        'Updates indexes before each epoch'

        self.indexes = np.arange(len(self.labels))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_labels_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels))
        y = []

        # Generate data
        for i, label_path in enumerate(list_labels_temp):
            # Store sample

            img_file = label_path.split("/")[-1].split("\\")[-1].replace(".txt", "")
            img_path = self.imgs_dir+"/" + img_file + ".jpg"
            img_paths = glob(os.path.join(img_path))
#             print("img_path:",img_path)
            img = cv2.imread(img_paths[0])
            # Store class
            pts_label = self.readShapes(label_path)
            
#             plt.imshow(img)
#             plt.show()
            
            augment_img, pts_label = self.process_data_item(img, pts_label[0], self.dim, self.model_stride)

            X[i,] = augment_img
#             print("pts_label",np.shape(pts_label))
            y.append(pts_label)

#             plt.imshow(augment_img)
#             plt.show() 
            
#             print("YY shape,",np.shape(pts_label))    
#             plt.imshow(pts_label[:,:,0]*255 )
#             plt.show()

        return np.asarray(X), np.asarray(y)

    def readShapes(self, path):
        shapes = []
        with open(path) as fp:
            for line in fp:
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
        
        image_size = np.shape(XX)[0]
#         pts2 = np.transpose(np.asarray(pts))

#         pts2 = pts2.reshape((-1,1,2))*image_size
#         pts2 = np.asarray(pts2 ,np.int32)  
#         cv2.polylines(XX,[np.asarray([pts2[0],pts2[3],pts2[2],pts2[1]],np.int32)],True,(0,255,255),2)


        YY = labels2output_map(llp, pts, dim, model_stride)
         
#         print("YY shape,",np.shape(YY))    
#         plt.imshow(YY[:,:,0]*255 )
#         plt.show()
        
        
        return XX, YY

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