# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,classification_report,roc_auc_score,matthews_corrcoef,mean_absolute_error,mean_squared_error,root_mean_squared_error,mean_absolute_percentage_error,explained_variance_score,r2_score,roc_curve,precision_recall_curve,auc
from sklearn.cluster import k_means,DBSCAN

