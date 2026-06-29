import json 
import streamlit as st 
import numpy as np 
from PIL import Image
import tensorflow as tf 

with open("classes.json","r") as file:
    my_list=json.load(file)
print(my_list)

def load_model():
    model=tf.keras.models.load_model(''
    'model_train.hdf5')
    return model
def predict_class(image,model):
    image=tf.cast(image,tf.float32)
    image=tf.image.resize(image,[256,256])
    image=np.expand_dims(image,axis=0)
    prediction=model.predict(image)
    return prediction

model=load_model()
file=st.file_uploader(" ",type=["jpg","png"])


if file is None:
    st.text("waiting to upload")
else:
    slot=st.empty()
    test_image=Image.open(file)
    st.image(test_image,caption="Input image",width=400)
    pred=predict_class(np.asarray(test_image),model)
    class_names=my_list
    result=class_names[np.argmax(pred)]
    output="the image is "+result
    slot.text("done")
    st.success(output)
    print(result)



