import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
num_class=4 #5 class là 5 loài hoa 
            #2 class là chó/mèo

img_width=224
img_height=224

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (img_width, img_height),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('test',
                                            target_size = (img_width, img_height),
                                            batch_size = 32,
                                            class_mode = 'categorical')



def createModel():
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[img_width, img_height, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    opt = Adam(lr=0.001) #thêm vài số 0 vào nếu accuracy ko tăng :v
    cnn.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return cnn


cnn = createModel()
#cnn.load_weights('best_model.h5')
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
history=cnn.fit(x = training_set, validation_data = test_set, epochs =  20,callbacks=[checkpoint,early])
cnn.save('breast_model.h5', save_format="h5")

model = load_model('breast_model.h5')
print('Evaluate Model .....')
score = model.evaluate(test_set)

print('Test Lost : ',score[0])
print('Test Accuracy : ',score[1])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('a.pdf')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('b.pdf')
plt.show()

#from keras.models import load_model
#import cv2
#import numpy as np
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
#model = load_model('breast_model_new.h5')

#imgs=[]

#class_names = ['benign','insitu','invasive','normal']
#img = cv2.imread('be1.tif')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (224, 224))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('is1.tif')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (224, 224))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('iv1.tif')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (224, 224))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('n1.tif')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (224, 224))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('tulip1.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (64, 64))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)


#class_names = ['cat','dog']
#img = cv2.imread('cat1.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (64, 64))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('cat2.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (64, 64))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('dog1.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (64, 64))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)
#img = cv2.imread('dog2.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (64, 64))
#img = img_to_array(img)
#img = preprocess_input(img)
#imgs.append(img)

#imgs = np.array(imgs, dtype="float32")
#classes = model.predict_classes(imgs,batch_size=32)
#res = model.predict(imgs)

#re_class=[]

#for c in classes :
#    a=class_names[c] 
#    re_class.append(a)

#print(re_class)
#print("\n")
#for pred in res:
#    i=0;
#    print(re_class[i])
#    for class_name in class_names:   
#        print("%s: %.10f%%" % (class_name, pred[i]*100))
#        i+=1
#    print("\n")
        
