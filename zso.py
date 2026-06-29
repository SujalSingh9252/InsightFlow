import os
import shutil
import zipfile
import streamlit as st

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix,classification_report,ConfusionMatrixDisplay
import streamlit as st
import zipfile
import os
from PIL import Image,ImageOps
import io
import shutil
import ast
import json
import time
import webbrowser

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras import Model 
from tensorflow.data.experimental import cardinality
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.applications import ResNet50,VGG19,VGG16
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img

# Function to count subfolders




def visualize_training_results(results):
                history = results.history
                
                # Plot Loss
                fig1, ax1 = plt.subplots()
                ax1.plot(history['val_loss'], label='val_loss')
                ax1.plot(history['loss'], label='loss')
                ax1.legend()
                ax1.set_title('Loss')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss')
                st.pyplot(fig1)  # Streamlit function to display the plot
                
                # Plot Accuracy
                fig2, ax2 = plt.subplots()
                ax2.plot(history['val_accuracy'], label='val_accuracy')
                ax2.plot(history['accuracy'], label='accuracy')
                ax2.legend()
                ax2.set_title('Accuracy')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Accuracy')
                st.pyplot(fig2)  # Streamlit function to display the plot

def plot_confusion_matrix(true_classes, y_pred):
    fig, ax = plt.subplots()
    final_disp = ConfusionMatrixDisplay.from_predictions(true_classes, y_pred, 
                                                        
                                                         xticks_rotation="vertical", 
                                                         cmap=plt.cm.Blues, 
                                                         ax=ax)
    st.pyplot(fig)               
                
    
  
def count_subfolders(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def extract_images(zip_file, extract_to_folder):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
        return zip_ref.namelist()

st.title('automatic deep learning')

delete_checkbox = st.sidebar.checkbox('Delete data after use')

extracted_folder = "extracted_images"
new_dir = 'data/split'

if delete_checkbox:
    delete_folder(extracted_folder)
    delete_folder(new_dir)
    st.sidebar.write("Data deleted after use!")


uploaded_zip = st.file_uploader("Upload a ZIP file containing images", type=["zip"])

if st.sidebar.button("Prediction", key="dashboard_btn"):
    with st.spinner('Opening Dashboard...'):
        import subprocess
        import webbrowser
        import time
        import threading
        with st.spinner('Opening Dashboard...'):
            def open_browser():
                time.sleep(2)  # Allow time for Streamlit to start
                webbrowser.open("http://localhost:8503")  # Open the app in browser

        # Start the browser opening in a separate thread
        threading.Thread(target=open_browser).start()

        # Run the Streamlit app on port 8503
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8503"])


if uploaded_zip:
    # Create folder for extracted images
    os.makedirs(extracted_folder, exist_ok=True)

    # Extract images from the ZIP file
    extracted_files = extract_images(uploaded_zip, extracted_folder)

    # Filter image files
    image_files = [f for f in extracted_files if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    # st.write(f"Number of images in the uploaded file: {len(image_files)}")

    # Count subfolders
    subfolders = count_subfolders(extracted_folder)
    # st.write(subfolders)
    folder_names = [os.path.basename(subfolder) for subfolder in subfolders]
    st.write("Classes:")
    st.write(folder_names)
    with open("classes.json","w") as file:
        json.dump(folder_names,file)
    if(len(folder_names)>2):
        act='softmax'
        lo=CategoricalCrossentropy()
    
    else:
        act='sigmoid'
        lo='binary_crossentropy'
    

    # Create train and test folders
    os.makedirs(new_dir, exist_ok=True)
    train_folder = os.path.join(new_dir, 'train')
    test_folder = os.path.join(new_dir, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Split images into train/test

    train_ratio = 0.8
    for i, subfolder in enumerate(subfolders):
        img_files = [f for f in os.listdir(subfolder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        train_size = int(train_ratio * len(img_files))
        train_subfolder = os.path.join(train_folder, folder_names[i])
        test_subfolder = os.path.join(test_folder, folder_names[i])
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(test_subfolder, exist_ok=True)

        # Copy train images
        for img in img_files[:train_size]:
            shutil.copyfile(os.path.join(subfolder, img), os.path.join(train_subfolder, img))

        # Copy test images
        for img in img_files[train_size:]:
            shutil.copyfile(os.path.join(subfolder, img), os.path.join(test_subfolder, img))

    st.write("Images have been split into train and test")


    train_folder='data/split/train'
    test_folder='data/split/test'
    train_gen=ImageDataGenerator(rescale=1./255,validation_split=0.125)
    test_gen=ImageDataGenerator(rescale=1./255)

    train_generator=train_gen.flow_from_directory(train_folder,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    batch_size=32,
                                                    shuffle=True,
                                                    seed=42)
    
    val_generator=train_gen.flow_from_directory(train_folder,class_mode="categorical",
                                                    subset='validation',
                                                    batch_size=32,
                                                    shuffle=True,
                                                    seed=42)
    
    test_generator=test_gen.flow_from_directory(test_folder,
                                                class_mode='categorical',
                                                batch_size=32,
                                                shuffle=False,
                                                seed=42)
    

    # creating datasets
    train_images,train_labels=next(train_generator)
    test_images,test_labels=next(test_generator)
    val_images,val_labels=next(val_generator)

    train_classes=train_generator.classes
    val_classes=val_generator.classes
    test_classes=test_generator.classes


    train_class,train_count=np.unique(train_classes,return_counts=True)
    val_class,val_count=np.unique(val_classes,return_counts=True)
    test_class,test_count=np.unique(test_classes,return_counts=True)


    print('Train ~ {}'.format(list(zip(train_class,train_count))))
    print('validation ~ {}'.format(list(zip(val_class,val_count))))
    print('Test ~ {}'.format(list(zip(test_class,test_count))))

    
    train_class_names=train_generator.class_indices
    st.write("Train:",train_class_names)
    val_class_names=train_generator.class_indices
    st.write("Validation:",val_class_names)
    test_class_names=test_generator.class_indices
    st.write("Test:",test_class_names)


    # st.write("train shape")
    # st.write(np.shape(train_images))
    # st.write(np.shape(train_labels))

    # st.write("validation")
    # st.write(np.shape(val_images))
    # st.write(np.shape(val_labels))

    # st.write("test")
    # st.write(np.shape(test_images))
    # st.write(np.shape(test_labels))

    num_classes = len(set(train_generator.classes))
    # st.write(num_classes)


    list_of_model=["basic","VGG19","ResNet50","VGG16"]

    modeling=st.selectbox("select a model",list_of_model)
    if(modeling=='basic'):
        baseline_model=Sequential()
        baseline_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same',input_shape=(256,256,3)))
        baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
        baseline_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
        baseline_model.add(MaxPooling2D(pool_size=(2, 2)))


# Layer 2- connect all nodes with dense layer
        baseline_model.add(Flatten())
        baseline_model.add(Dense(128,activation='relu'))
        baseline_model.add(Dense(64, activation='relu'))

        # Output Layer- softmax activiation for multi-categorical with 10 classes
        baseline_model.add(Dense(num_classes, activation=act))

        #Compile the sequential CNN model- adam optimizer, 
        # categorical_crossentropy loss, and set our metric to accuracy
    

        # print model summary
        baseline_model.summary()
        #Fit the model 
    if(modeling=="VGG19"):
        vgg19=VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=(256,256,3))
        vgg19.summary()
        baseline_model=Sequential()
        baseline_model.add(vgg19)
        baseline_model.add(layers.Flatten())
        baseline_model.add(layers.Dense(64,activation='relu'))
        baseline_model.add(layers.Dense(num_classes,activation=act))
        baseline_model.summary()
    if(modeling=="ResNet50"):
        rn_50=ResNet50(weights='imagenet',
                                include_top=False,
                                input_shape=(256,256,3))
        baseline_model=Sequential()
        baseline_model.add(rn_50)
        baseline_model.add(layers.Flatten())
        baseline_model.add(layers.Dense(64,activation='relu'))
        baseline_model.add(layers.Dense(num_classes,activation=act))
        baseline_model.summary()
    if(modeling=="VGG16"):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        for layer in base_model.layers:
            layer.trainable = False
        
        baseline_model=Sequential()
        baseline_model.add(base_model)
        baseline_model.add(Flatten())
        baseline_model.add(Dense(256,activation='relu'))
        baseline_model.add(Dense(128,activation='relu'))
        baseline_model.add(Dense(64,activation='relu'))
  
        baseline_model.add(Dense(num_classes,activation=act))
        baseline_model.summary()
        buffer = io.StringIO()
        baseline_model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        st.text(buffer.getvalue())
    
try:
    epo = st.text_input("Enter 'no' of epochs:")
    epoc=int(epo)
    baseline_model.compile(optimizer='adam', 
                            loss=lo,  
                            metrics=['accuracy'])
    # modeling_start=st.checkbox("start training")
    if st.button("start training"):
    # Training the model
        baseline_history = baseline_model.fit(
            train_generator,
            epochs=epoc,
            batch_size=32,
            verbose=1,
            validation_data=val_generator,
        )
        
        # Evaluate the model
        test_loss, test_acc = baseline_model.evaluate(test_generator, verbose=1)
        
        # Display test results
        st.write('Test Loss: ', test_loss)
        st.write('Test Accuracy: ', test_acc)

        # Visualize training results
        visualize_training_results(baseline_history)
        predictions = baseline_model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        true_classes=test_generator.classes
        plot_confusion_matrix(true_classes,y_pred
        )



        # Save the model
        save_model(baseline_model, "model_train.hdf5")
        st.write("Model saved successfully!")
except:

        st.write("")            
