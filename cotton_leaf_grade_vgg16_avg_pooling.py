####This VGG16 model is used to train and classify cotton leaf grades.######################################################
###leaf grades used here are 2,3,4, 5 and 6.
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model, Input
from keras.optimizers import Adadelta, RMSprop, SGD
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D, concatenate, Add
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from itertools import product
from PIL import Image


Name="contaminant_grade{}".format(int(time.time()))
tensorboard=TensorBoard(log_dir='logs/{}'.format(Name))


image_path = pd.read_csv('/content/drive/MyDrive/contaminant_grade_VGG16/TAMU_Bale_Info.csv', sep=",", header=None)
image_pathDF=pd.DataFrame(image_path)
grade_list=[]
for i in range(1,575):
  grade1=image_pathDF.iloc[i,7]
  grade2=pd.to_numeric(grade1)
  grade_list.append(grade2)
  
class2_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==2:
    class2_img=image_pathDF.iloc[k+1,0]
    head2, tail2 = os.path.split(class2_img)
    class2_imglist.append(tail2)

class3_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==3:
    class3_img=image_pathDF.iloc[k+1,0]
    head3, tail3 = os.path.split(class3_img)
    class3_imglist.append(tail3)

class4_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==4:
    class4_img=image_pathDF.iloc[k+1,0]
    head4, tail4 = os.path.split(class4_img)
    class4_imglist.append(tail4)

class5_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==5:
    class5_img=image_pathDF.iloc[k+1,0]
    head5, tail5 = os.path.split(class5_img)
    class5_imglist.append(tail5)

class6_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==6:
    class6_img=image_pathDF.iloc[k+1,0]
    head6, tail6 = os.path.split(class6_img)
    class6_imglist.append(tail6)

os.chdir('/content/drive/MyDrive/contaminant_grade_VGG16/Pics_for_TAMU')
!chmod +x /content/drive/MyDrive/contaminant_grade_VGG16/Pics_for_TAMU
files = [f for f in os.listdir('.') if os.path.isfile(f)]

####################We will now split each image into 416x416############################Run this section only the first time
# for fp in class2_imglist:
#   img = Image.open(fp)
#   d=416
#   w,h = img.size
#   image_num=1  
#   grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
#   for i, j in grid:
#     box = (j, i, j+d, i+d)
#     crop=img.crop(box)
#     savedir= "/content/drive/MyDrive/contaminant_grade_VGG16/class2_split"
#     name = os.path.basename(fp)
#     name = os.path.splitext(name)[0]
#     save_to= os.path.join(savedir, name+"_{:03}.tif")
#     crop.save(save_to.format(image_num))
#     image_num += 1

# for fp in class3_imglist:
#   img = Image.open(fp)
#   d=416
#   w,h = img.size
#   image_num=1  
#   grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
#   for i, j in grid:
#     box = (j, i, j+d, i+d)
#     crop=img.crop(box)
#     savedir= "/content/drive/MyDrive/contaminant_grade_VGG16/class3_split"
#     name = os.path.basename(fp)
#     name = os.path.splitext(name)[0]
#     save_to= os.path.join(savedir, name+"_{:03}.tif")
#     crop.save(save_to.format(image_num))
#     image_num += 1

# for fp in class4_imglist:
#   img = Image.open(fp)
#   d=416
#   w,h = img.size
#   image_num=1  
#   grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
#   for i, j in grid:
#     box = (j, i, j+d, i+d)
#     crop=img.crop(box)
#     savedir= "/content/drive/MyDrive/contaminant_grade_VGG16/class4_split"
#     name = os.path.basename(fp)
#     name = os.path.splitext(name)[0]
#     save_to= os.path.join(savedir, name+"_{:03}.tif")
#     crop.save(save_to.format(image_num))
#     image_num += 1
                        
# for fp in class5_imglist:
#   img = Image.open(fp)
#   d=416
#   w,h = img.size
#   image_num=1  
#   grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
#   for i, j in grid:
#     box = (j, i, j+d, i+d)
#     crop=img.crop(box)
#     savedir= "/content/drive/MyDrive/contaminant_grade_VGG16/class5_split"
#     name = os.path.basename(fp)
#     name = os.path.splitext(name)[0]
#     save_to= os.path.join(savedir, name+"_{:03}.tif")
#     crop.save(save_to.format(image_num))
#     image_num += 1
        
# for fp in class6_imglist:
#   img = Image.open(fp)
#   d=416
#   w,h = img.size
#   image_num=1  
#   grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
#   for i, j in grid:
#     box = (j, i, j+d, i+d)
#     crop=img.crop(box)
#     savedir= "/content/drive/MyDrive/contaminant_grade_VGG16/class6_split"
#     name = os.path.basename(fp)
#     name = os.path.splitext(name)[0]
#     save_to= os.path.join(savedir, name+"_{:03}.tif")
#     crop.save(save_to.format(image_num))
#     image_num += 1
##################Image splitting complete###################################################################

#Loading Training Data Set
image_data_list2=[]
for file in os.listdir('/content/drive/MyDrive/contaminant_grade_VGG16/class2_split'):
    full_path='/content/drive/MyDrive/contaminant_grade_VGG16/class2_split/'+ str(file)
    head, tail= os.path.split(full_path)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list2.append(image)

image_data_list3=[]
for file in os.listdir('/content/drive/MyDrive/contaminant_grade_VGG16/class3_split'):
    full_path='/content/drive/MyDrive/contaminant_grade_VGG16/class3_split/'+ str(file)
    head, tail= os.path.split(full_path)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list3.append(image)

image_data_list4=[]
for file in os.listdir('/content/drive/MyDrive/contaminant_grade_VGG16/class4_split'):
    full_path='/content/drive/MyDrive/contaminant_grade_VGG16/class4_split/'+ str(file)
    head, tail= os.path.split(full_path)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list4.append(image)

image_data_list5=[]
for file in os.listdir('/content/drive/MyDrive/contaminant_grade_VGG16/class5_split'):
    full_path='/content/drive/MyDrive/contaminant_grade_VGG16/class5_split/'+ str(file)
    head, tail= os.path.split(full_path)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list5.append(image)

image_data_list6=[]
for file in os.listdir('/content/drive/MyDrive/contaminant_grade_VGG16/class6_split'):
    full_path='/content/drive/MyDrive/contaminant_grade_VGG16/class6_split/'+ str(file)
    head, tail= os.path.split(full_path)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list6.append(image)
   
image_data_list=[]
for i in range(len(image_data_list2)):
  image_data_list.append(image_data_list2[i])
for i in range(len(image_data_list3)):
  image_data_list.append(image_data_list3[i])
for i in range(len(image_data_list4)):
  image_data_list.append(image_data_list4[i])
for i in range(len(image_data_list5)):
  image_data_list.append(image_data_list5[i])
for i in range(len(image_data_list6)):
  image_data_list.append(image_data_list6[i])



image_data=np.array(image_data_list)
image_data=np.rollaxis(image_data,1,0)
image_data=image_data[0] 

#Defining number of classes
num_classes=5
num_samples=image_data.shape[0]
labels=np.ones(num_samples,dtype='int64')
labels[0:42]=0      # first 244 images are class'0'
labels[42:727]=1    #next 132 images are class '1'
labels[727:1960]=2  # next 293 images are class '2'
labels[1960:2496]=3  #next 140 images are class '3'
labels[2496:2508]=4  #next 2 images are class '4'
names=['lf-2','lf-3', 'lf-4','lf-5', 'lf-6']
#convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the training data set
xtrain,ytrain=shuffle(image_data,Y,random_state=2)
X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, train_size=0.80,test_size=0.20,random_state=2)

#Using VGG16 as Feature Extraxtor and then as a classifier
model1=VGG16(include_top=False,weights='imagenet') #as feature extractor
model2 = VGG16(include_top=True,weights='imagenet',input_tensor=Input(shape=(224,224, 3)))  #as a classifier

#since we have only 5 classes and not 1000 classes, we replace the last layer..
#..by our own Dense layer named 'prediction' with 2 classes
b1conv2=model2.get_layer('block1_conv2').output
skip1=AveragePooling2D(pool_size=(224,224),strides=None,padding='valid')(b1conv2)
dense1=Dense(64,name='dense1',activation='relu',trainable=True)(skip1)
b2conv1=model2.get_layer('block2_conv1')(dense1)

b2conv2=model2.get_layer('block2_conv2').output
skip2=AveragePooling2D(pool_size=(112,112),strides=None,padding='valid')(b2conv2)
dense2=Dense(128,name='dense2',activation='relu',trainable=True)(skip2)
b3conv1=model2.get_layer('block3_conv1')(dense2)

b3conv2=model2.get_layer('block3_conv3').output
skip3=AveragePooling2D(pool_size=(56,56),strides=None,padding='valid')(b3conv2)
dense3=Dense(256,name='dense3',activation='relu',trainable=True)(skip3)
b4conv1=model2.get_layer('block4_conv1')(dense3)

b4conv2=model2.get_layer('block4_conv3').output
skip4=AveragePooling2D(pool_size=(28,28),strides=None,padding='valid')(b4conv2)
dense4=Dense(512,name='dense4',activation='relu',trainable=True)(skip4)
b5conv1=model2.get_layer('block5_conv1')(dense4)


second_last_layer=model2.get_layer('block5_pool').output
Avg2D_layer=AveragePooling2D(pool_size=(7,7),strides=None,padding='valid')(second_last_layer)
last_layer=model2.get_layer('flatten')(Avg2D_layer)
out1=Dense(128,name='ThirdLast',activation='relu',trainable=True)(last_layer)
out2=Dense(128,name='SecondLast',activation='relu',trainable=True)(out1)
out3=Dense(num_classes,name='LeafGradePrediction',activation='softmax',trainable=True)(out2)
#out=Dense(num_classes,activation='softmax')(last_layer)
cotton_grade_model=Model(inputs=model2.input,outputs=out3)

#Freezing all the layers except the last layer.The last yer is the only trainable layer
#for layer in squat_detect_model.layers[:-1]:
   # layer.trainable=False

opt = SGD(learning_rate=0.0001,momentum=0.09, decay=0.01)
cotton_grade_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
t=time.time()

#for layer in cotton_grade_model.layers[:-6]:     #Training just last 6 layers
# layer.trainable=False


trained_data=cotton_grade_model.fit(X_train,y_train, batch_size=4, epochs=30, verbose=1, validation_data=(X_test,y_test), callbacks=[tensorboard])
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = cotton_grade_model.evaluate(X_test,y_test, batch_size=2, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


temp=cotton_grade_model.predict(X_test)
print('accuracy score is =',accuracy_score(y_test.argmax(axis=1),temp.argmax(axis=1))*100,'%')
print(confusion_matrix(y_test.argmax(axis=1), temp.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1),temp.argmax(axis=1)))
train_loss=trained_data.history['loss']
val_loss=trained_data.history['val_loss']
train_acc=trained_data.history['accuracy']
val_acc=trained_data.history['val_accuracy']
plt.figure(1,figsize=(7,5))
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.figure(2,figsize=(7,5))
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

#Now saving the pretrained weights
# serialize model to JSON
model_json = trained_data.model.to_json()
with open("trained_data_cotton_grade.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
trained_data.model.save("trained_data_cotton_grade.h5")
print("Saved trained_data to disk")    
