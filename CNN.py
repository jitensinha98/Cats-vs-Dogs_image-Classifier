import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

train_dir='train'
img_size=50

training_data=[]

nClasses = 2
n_epochs = 25

def label_image(img):
	word_label=img.split('.')[-3]
	if word_label == 'cat' :
		return [1,0]
	elif word_label == 'dog' :
		return [0,1]

def prep_dataset():
	for img in tqdm(os.listdir(train_dir)):	
		label=label_image(img)
		path=os.path.join(train_dir,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
		training_data.append([np.array(img),np.array(label)])	

	shuffle(training_data,random_state=23)

	train_data=training_data[:-500]
	test_data=training_data[-500:]

	train_X=np.array([i[0] for i in train_data]).reshape(-1,img_size,img_size,1)
	train_Y=np.array([i[1] for i in train_data])

	test_X=np.array([i[0] for i in test_data]).reshape(-1,img_size,img_size,1)
	test_Y=np.array([i[1] for i in test_data])

	return train_X,train_Y,test_X,test_Y


def Conv_Net(input_shape):
	model = Sequential()
	model.add(Conv2D(64, (2, 2), padding='same', activation='relu', input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))	
	model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
 
	model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
 
	model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
 
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nClasses, activation='softmax'))
     
	return model

train_X,train_Y,test_X,test_Y=prep_dataset()
model = Conv_Net(train_X.shape[1:])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
training_model = model.fit(train_X, train_Y,batch_size=128,epochs=n_epochs,verbose=1)

score = model.evaluate(test_X, test_Y, verbose=0)

print("Test Loss = ",score[0])
print("Test Accuracy = ",score[1])

model.save('Saved_Model/Cat_Dog_Classifier.h5')


	 
