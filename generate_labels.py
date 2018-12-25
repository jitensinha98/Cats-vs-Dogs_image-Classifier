import os
from keras.models import load_model
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.utils import shuffle
import pandas as pd

path='test'

img_size=50

test_data=[]

image_name=[]
labels=[]

model = load_model("Saved_Model/Cat_Dog_Classifier.h5")

print("")
print("Processing...")
print("")

for img in tqdm(os.listdir(path)):
	path1=os.path.join(path,img)
	img_num=img.split('.')[0]
	image=cv2.resize(cv2.imread(path1,cv2.IMREAD_GRAYSCALE),(img_size,img_size))	
	test_data.append([np.array(image),img_num])

for num,data in enumerate(tqdm(test_data)):
    img_num = data[1]
    img_data = data[0]
   
    reshaped_image = img_data.reshape(-1,img_size,img_size,1)
    model_out = model.predict(reshaped_image)
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
	
    image_name.append(img_num)
    labels.append(str_label)

labelled_dataset=pd.DataFrame({'ID':image_name,'ANIMAL':labels})

print("")
print("Processing Complete.")
print("")

print("Saving Dataset...")
labelled_dataset.to_csv('Test_label.csv',index=False)
print("Dataset Saved.")


	
	

