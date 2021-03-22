from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import Adadelta
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D, concatenate
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

Name="contaminant_grade{}".format(int(time.time()))
tensorboard=TensorBoard(log_dir='logs/{}'.format(Name))


image_path = pd.read_csv('/path of csv file containing information regarding two data classes Info.csv', sep=",", header=None)
image_pathDF=pd.DataFrame(image_path)
grade_list=[]
for i in range(1,575):                 #This value depends upon size of image data in csv file i.e. row number
  grade1=image_pathDF.iloc[i,6]        # 6 is column number containing information of class
  grade2=pd.to_numeric(grade1)
  grade_list.append(grade2)
  

class41_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==41:
    class41_img=image_pathDF.iloc[k+1,0]
    head41, tail41 = os.path.split(class41_img)
    class41_imglist.append(tail41)

class53_imglist=[]
for k in range(len(grade_list)):
  if grade_list[k]==53:
    class53_img=image_pathDF.iloc[k+1,0]
    head53, tail53 = os.path.split(class53_img)
    class53_imglist.append(tail53)
        

os.chdir('image directory path')
!chmod +x /image directory path
#Loading Training Data Set
image_data_list=[]
for file in os.listdir('/image directory path'):
    full_path='/image directory path/'+ str(file)
    head, tail= os.path.split(full_path)
    for i in class41_imglist:
      if i==tail:
        print(full_path)
        image=load_img(full_path,target_size=(224,224))
        image=img_to_array(image)
        image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        image=preprocess_input(image)
        image_data_list.append(image)
    for j in class53_imglist:
      if j==tail:
        print(full_path)
        image=load_img(full_path,target_size=(224,224))
        image=img_to_array(image)
        image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        image=preprocess_input(image)
        image_data_list.append(image)

    
image_data=np.array(image_data_list)
image_data=np.rollaxis(image_data,1,0)
image_data=image_data[0] 

#Defining number of classes
num_classes=2
num_samples=image_data.shape[0]
labels=np.ones(num_samples,dtype='int64')
labels[0:244]=0  # first 244 images are class'0'
labels[244:369]=1#remaining 125 images are class '1'
names=['grade-41','grade-53']
#convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the training data set
xtrain,ytrain=shuffle(image_data,Y,random_state=2)
X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, train_size=0.80,test_size=0.20,random_state=2)

#Using VGG16 as Feature Extraxtor and then as a classifier
model1=VGG16(include_top=False,weights='imagenet') #as feature extractor
model2 = VGG16(include_top=True,weights='imagenet')  #as a classifier

#since we have only 2 classes and not 1000 classes, we replace the last layer..
#..by our own Dense layer named 'prediction' with 2 classes

second_last_layer=model2.get_layer('block5_pool').output
Avg2D_layer=AveragePooling2D(pool_size=(7,7),strides=None,padding='valid')(second_last_layer)
last_layer=model2.get_layer('flatten')(Avg2D_layer)
out1=Dense(4096,name='grade41',activation='relu',trainable=True)(last_layer)
out2=Dense(4096,name='grade53',activation='relu',trainable=True)(out1)
out3=Dense(num_classes,name='predict',activation='softmax',trainable=True)(out2)
#out=Dense(num_classes,activation='softmax')(last_layer)
cotton_grade_model=Model(inputs=model2.input,outputs=out3)

#Freezing all the layers except the last layer.The last yer is the only trainable layer
#for layer in squat_detect_model.layers[:-1]:
   # layer.trainable=False

opt = Adadelta(learning_rate=0.0001)
cotton_grade_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
t=time.time()
trained_data=cotton_grade_model.fit(X_train,y_train, batch_size=4, epochs=50, verbose=1, validation_data=(X_test,y_test), callbacks=[tensorboard])
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = cotton_grade_model.evaluate(X_test,y_test, batch_size=2, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#Now the customized model(plastic_model) has been trained for 12 epochs using 12 training images
#Training accuracy=100%, Validation accuracy = 33.33%. This means this model with perfomr with 33.33% accuracy on new data

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
