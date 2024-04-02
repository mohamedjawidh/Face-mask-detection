import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import classification_report
def print_image_shapes(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            print("Image shape:", img.shape)
        else:
            print("Error: Unable to load image at path:", img_path)
            train_directory = "F:/chandru/Face Mask Dataset/Train"
test_directory = "F:/chandru/Face Mask Dataset/Test"
validation_directory = "F:/chandru/Face Mask Dataset/Validation"
train_filepath = []
train_label = []
folds_train = os.listdir(train_directory)
for fold in folds_train:
    f_path = os.path.join(train_directory, fold)
    imgs = os.listdir(f_path)
    for img in imgs:
        img_path = os.path.join(f_path, img)
        train_filepath.append(img_path)
        train_label.append(fold)
        train_file_path_series = pd.Series(train_filepath, name='filepath')
train_label_path_series = pd.Series(train_label, name='label')
df_train = pd.concat([train_file_path_series, train_label_path_series], axis=1)
test_filepath = []
test_label = []

folds_test = os.listdir(test_directory)

for fold in folds_test:
    f_path = os.path.join(test_directory, fold)
    imgs = os.listdir(f_path)
    for img in imgs:
        img_path = os.path.join(f_path, img)
        test_filepath.append(img_path)
        test_label.append(fold)

test_file_path_series = pd.Series(test_filepath, name='filepath')
test_label_path_series = pd.Series(test_label, name='label')
df_test = pd.concat([test_file_path_series, test_label_path_series], axis=1)
validation_filepath = []
validation_label = []

folds_validation = os.listdir(validation_directory)

for fold in folds_validation:
    f_path = os.path.join(validation_directory, fold)
    imgs = os.listdir(f_path)
    for img in imgs:
        img_path = os.path.join(f_path, img)
        validation_filepath.append(img_path)
        validation_label.append(fold)


validation_file_path_series = pd.Series(validation_filepath, name='filepath')
validation_label_path_series = pd.Series(validation_label, name='label')
df_validation = pd.concat([validation_file_path_series, validation_label_path_series], axis=1)
rescale = layers.Rescaling(1./255)


train_ds = keras.utils.image_dataset_from_directory(
    directory='F:/chandru/Face Mask Dataset/Train',
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',   
)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))  


validation_ds = keras.utils.image_dataset_from_directory(
    directory='F:/chandru/Face Mask Dataset/Validation',
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode='categorical',  
)
validation_ds = validation_ds.map(lambda x, y: (rescale(x), y))  


test_ds = keras.utils.image_dataset_from_directory(
    directory='F:/chandru/Face Mask Dataset/Test',
    batch_size=32,
    image_size=(224, 224),
    label_mode='categorical',  
    shuffle=False,
)
test_ds = test_ds.map(lambda x, y: (rescale(x), y)) 
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))


model = tf.keras.models.Sequential([
    input_layer,
    
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    
    tf.keras.layers.MaxPooling2D(),
   
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    
    tf.keras.layers.MaxPooling2D(),
   
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
   
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dropout(0.5),
   
    tf.keras.layers.Dense(2, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)


history = model.fit(train_ds,
                    validation_data=validation_ds,
                    epochs=5,
                    callbacks=[early_stopping])
test_loss, test_accuracy = model.evaluate(test_ds) 


print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
test_images = ['1114.png','1504.jpg', '0072.jpg','0012.jpg','0353.jpg','1374.jpg']

gamma = 2.0
fig = plt.figure(figsize = (14,14))
rows = 3
cols = 2
axes = []
assign = {'0':'Mask','1':"No Mask"}
for j,im in enumerate(test_images):
    image =  cv2.imread(os.path.join(image_directory,im),1)
    image =  adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                im = cv2.resize(frame,(img_size,img_size))
                im = np.array(im)/255.0
                im = im.reshape(1,124,124,3)
                result = model.predict(im)
                if result>0.5:
                    label_Y = 1
                else:
                    label_Y = 0
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image,assign[str(label_Y)] , (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
        
        except:pass
    axes.append(fig.add_subplot(rows, cols, j+1))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

model = Sequential()

model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(124,124,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam' ,metrics=['accuracy'])
xtrain,xval,ytrain,yval=train_test_split(X, Y,train_size=0.8,random_state=0)
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,    
        rotation_range=15,    
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=True,  
        vertical_flip=False)
datagen.fit(xtrain)

history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=32),
                    steps_per_epoch=xtrain.shape[0]//32,
                    epochs=50,
                    verbose=1,
                    validation_data=(xval, yval))
X = []
Y = []
for features,label in data:
    X.append(features)
    Y.append(label)

X = np.array(X)/255.0
X = X.reshape(-1,124,124,3)
Y = np.array(Y)
ata = []
img_size = 124
mask = ['face_with_mask']
non_mask = ["face_no_mask"]
labels={'mask':0,'without mask':1}
for i in df["name"].unique():
    f = i+".json"
    for j in getJSON(os.path.join(directory,f)).get("Annotations"):
        if j["classname"] in mask:
            x,y,w,h = j["BoundingBox"]
            img = cv2.imread(os.path.join(image_directory,i),1)
            img = img[y:h,x:w]
            img = cv2.resize(img,(img_size,img_size))
            data.append([img,labels["mask"]])
        if j["classname"] in non_mask:
            x,y,w,h = j["BoundingBox"]
            img = cv2.imread(os.path.join(image_directory,i),1)
            img = img[y:h,x:w]
            img = cv2.resize(img,(img_size,img_size))    
            data.append([img,labels["without mask"]])
random.shuffle(data)

p = []
for face in data:
    if(face[1] == 0):
        p.append("Mask")
    else:
        p.append("No Mask")
sns.countplot(p)
{'FileName': '2349.png',
 'NumOfAnno': 4,
 'Annotations': [{'isProtected': False,
   'ID': 193452793312540288,
   'BoundingBox': [29, 69, 285, 343],
   'classname': 'face_other_covering',
   'Confidence': 1,
   'Attributes': {}},
  {'isProtected': False,
   'ID': 545570408121800384,
   'BoundingBox': [303, 99, 497, 341],
   'classname': 'face_other_covering',
   'Confidence': 1,
   'Attributes': {}},
  {'isProtected': False,
   'ID': 339053397051370048,
   'BoundingBox': [8, 71, 287, 373],
   'classname': 'hijab_niqab',
   'Confidence': 1,
   'Attributes': {}},
  {'isProtected': False,
   'ID': 100482004994698944,
   'BoundingBox': [296, 99, 525, 371],
   'classname': 'hijab_niqab',
   'Confidence': 1,
   'Attributes': {}}]}
cvNet = cv2.dnn.readNetFromCaffe('weights.caffemodel')
def getJSON(filePathandName):
    with open(filePathandName,'r') as f:
        return json.load(f)
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))