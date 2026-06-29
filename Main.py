import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import plotly.express as px 
import os 
import warnings
import datetime
import plotly.graph_objects as go 
import numpy as np
import seaborn as sns
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.stats as stats
from PIL import Image 
import joblib
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from Models import ClassificationMetrics, RegressionMetrics
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,classification_report,roc_auc_score,matthews_corrcoef,mean_absolute_error,mean_squared_error,root_mean_squared_error,mean_absolute_percentage_error,explained_variance_score,r2_score,roc_curve,precision_recall_curve


def create_heatmap(df):
     
     df_encoded = df.copy()
     label_encoder = LabelEncoder()
     for column in df_encoded.columns:
      df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
     corr_matrix=df_encoded.corr()
     heatmap=ff.create_annotated_heatmap(
          z=corr_matrix.values,
          x=list(corr_matrix.columns),
          y=list(corr_matrix.index),
          annotation_text=corr_matrix.round(2).values,
          colorscale="Viridis",
          showscale=True
     )
     heatmap.update_layout(
          title="correaltion heatmap",
          xaxis_title="Features",
          template="plotly_white"
     )
     return heatmap
page_icon = Image.open("logo.png")
st.set_page_config(layout="wide", page_title="Machine Learning Studio", page_icon=page_icon)
def initial_state():
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    if 'X_train' not in st.session_state:
        st.session_state['X_train'] = None

    if 'X_test' not in st.session_state:
        st.session_state['X_test'] = None

    if 'y_train' not in st.session_state:
        st.session_state['y_train'] = None

    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = None

    if 'X_val' not in st.session_state:
        st.session_state['X_val'] = None

    if 'y_val' not in st.session_state:
        st.session_state['y_val'] = None

    if 'model' not in st.session_state:
        st.session_state['model']=None 
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model']=False
    if 'trained_model_bool' not in st.session_state:
        st.session_state['trained_model_bool'] = False
    if  "problem_type" not in st.session_state:
        st.session_state['problem_type']=False 
    if  "metrics_df" not in st.session_state:
        st.session_state['metrics_df']=pd.DataFrame()
    if "is_train" not in st.session_state:
        st.session_state['is_train'] = False

    if "is_test" not in st.session_state:
        st.session_state['is_test'] = False

    if "is_val" not in st.session_state:
        st.session_state['is_val'] = False

    if "show_eval" not in st.session_state:
        st.session_state['show_eval'] = False

    if "all_the_process" not in st.session_state:
        st.session_state['all_the_process'] = """"""

    if "all_the_process_predictions" not in st.session_state:
        st.session_state['all_the_process_predictions'] = False

    if 'y_pred_train' not in st.session_state:
        st.session_state['y_pred_train'] = None

    if 'y_pred_test' not in st.session_state:
        st.session_state['y_pred_test'] = None

    if 'y_pred_val' not in st.session_state:
        st.session_state['y_pred_val'] = None

    if 'uploading_way' not in st.session_state:
        st.session_state['uploading_way'] = None

    if "lst_models" not in st.session_state:
        st.session_state["lst_models"] = []

    if "lst_models_predictions" not in st.session_state:
        st.session_state["lst_models_predictions"] = []

    if "models_with_eval" not in st.session_state:
        st.session_state["models_with_eval"] = {}

    if "reset_1" not in st.session_state:
        st.session_state["reset_1"] = False

    
initial_state()
def new_line(n=1):
    for i in range(n):
        st.write("\n")
st.cache_data()
def load_data(upd_file):
    df=pd.read_csv(upd_file)
    return df 

def progress_bar():
    my_bar=st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete+1)
col1, col2 = st.columns([3, 6])
with col1:
    st.image("logo.png", width=200)
with col2:
    st.markdown(
        '<h1 class="main-header" style="font-family: Algerian;font-weight:400;">Machine Learning Studio</h1>',
        unsafe_allow_html=True,
    )

col1,col2,col3=st.columns([0.25,1,0.25])

new_line(2)

uploading_way=st.session_state.uploading_way
col1,col2,col3=st.columns(3,gap='large')


def upload_click():st.session_state.uploading_way="upload"
col1.markdown("<h5 align='center'> Upload file",unsafe_allow_html=True)
col1.button("Upload File",key="upload_file",use_container_width=True,on_click=upload_click)


        
# URL
def url_click(): st.session_state.uploading_way = "url"
col3.markdown("<h5 align='center'> Write URL", unsafe_allow_html=True)
col3.button("Write URL", key="write_url", use_container_width=True, on_click=url_click)

if st.session_state.df is None:
    if uploading_way=="upload":
        uploaded_file=st.file_uploader("upload dataset",type='csv')
        if uploaded_file:
            df=load_data(uploaded_file)
            st.session_state.df=df
   
    elif uploading_way=='url':
        url=st.text_input("Enter Url")
        if url:
            df=load_data(url)
            st.session_state.df=df 
if st.session_state.df is not None:
    df=st.session_state.df
    X_train=st.session_state.X_train
    X_test=st.session_state.X_test
    y_train=st.session_state.y_train
    y_test=st.session_state.y_test
    X_val=st.session_state.X_val
    y_val=st.session_state.y_val
    trained_model=st.session_state.trained_model
    is_train=st.session_state.is_train
    is_test=st.session_state.is_test
    is_val=st.session_state.is_val
    model=st.session_state.model 
    show_eval=st.session_state.show_eval
    y_pred_train=st.session_state.y_pred_train
    y_pred_test=st.session_state.y_pred_test
    y_pred_val=st.session_state.y_pred_val
    metrics_df=st.session_state.metrics_df

    st.divider()
    new_line()
    st.markdown(" ### Exploratory Data Analysis",unsafe_allow_html=True)
    new_line()
    with st.expander("EDA"):
        new_line()
        
        st.dataframe(st.session_state.df)
        col4,col5=st.columns([0.25,0.65])
        with col4:
            
            st.write(f"This DataFrame has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
            new_line()
        with col5:
           
            
            st.write(pd.DataFrame(df.columns, columns=['Columns']).T)
            new_line()
       
            st.dataframe(df.describe(), use_container_width=True)
            new_line()

       
            if df.select_dtypes(include=object).columns.tolist():
                st.dataframe(df.describe(include=['object']), use_container_width=True)
                new_line()
            else:
                st.info("There is no Categorical Features.")
                new_line()
        
            fig=create_heatmap(df)
            st.plotly_chart(fig)
            new_line()
        


        col1, col2 = st.columns([0.4,1])
        with col1:
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False),height=350, use_container_width=True)

        with col2:
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values = df.isnull().sum()
                null_values = null_values[null_values > 0]
                null_values = null_values.sort_values(ascending=False)
                null_values = null_values.to_frame()
                null_values.columns = ['Count']
                null_values.index.names = ['Feature']
                null_values['Feature'] = null_values.index
                fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                st.plotly_chart(fig, use_container_width=True)

        new_line()
        delete = st.checkbox("Delete Columns", value=False)
        new_line()
        if delete:
            col_to_delete = st.multiselect("Select Columns to Delete", df.columns)
            new_line()
            
            col1, col2, col3 = st.columns([1,0.7,1])
            if col2.button("Delete", use_container_width=True):
                st.session_state.all_the_process += f"""
# Delete Columns
df.drop(columns={col_to_delete}, inplace=True)
\n """
                progress_bar()
                df.drop(columns=col_to_delete, inplace=True)
                st.session_state.df = df
                st.success(f"The Columns **`{col_to_delete}`** are Deleted Successfully!")


        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        if col2.button("Show DataFrame", use_container_width=True):
            st.dataframe(df, use_container_width=True)

    new_line()
    st.markdown(" ### Exploratory Missing values",unsafe_allow_html=True)
    with st.expander("Missing Values"):
        new_line()
        missing=False
        new_line()
        if missing:
            col1,col2=st.columns(2,gap='medium')
            with col1:
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False), height=300, use_container_width=True)
            with col2:
                st.markdown("<h6 align='center'> Percentage of Null Values", unsafe_allow_html=True)
                null_percentage = pd.DataFrame(round(df.isnull().sum()/df.shape[0]*100, 2))
                null_percentage.columns=['Percentage']
                null_percentage['Percentage']=null_percentage['Percentage'].map('{:.2f} %'.format)
                null_percentage=null_percentage.sort_values(by='Percentage',ascending=False)
                st.dataframe(null_percentage,height=300,use_container_width=True)
                

            col1,col2,col3=st.columns([0.1,1,0.1])
            with col2:
                new_line()
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values=df.isnull().sum()
                null_values=null_values[null_values>0]
                null_values=null_values.sort_values(ascending=False)
                null_values=null_values.to_frame()
                null_values.columns=['Count']
                null_values.index.names=['Feature']
                null_values['Feature']=null_values.index
                if not null_values.empty:
                    fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No null values to display.")
            
        col1,col2=st.columns(2)
        with col1:
            missing_df_cols=df.columns[df.isnull().any()].to_list()
            if missing_df_cols:
                add_opt=["Numerical feature","Categorical feature"]
            else:
                add_opt=[]
            fill_feat=st.multiselect("select features",add_opt+missing_df_cols ,  help="Select Features to fill missing values")
                
        with col2:
                strategy=st.selectbox("missing value imputation",["Select", "Drop Rows", "Drop Columns",  "Median Imputation", "Mode Imputation"], help="Select Missing Values Strategy")
            
        if fill_feat and strategy !="Select":

            new_line() 
            col1,col2,col3=st.columns([1,0.5,1])
            if col2.button("Apply",use_container_width=True,key="missing_apply"):
                progress_bar()

                if "Numerical feature" in fill_feat:
                    fill_feat.remove("Numerical feature")
                    fill_feat+=df.select_dtypes(include=np.number).columns.to_list()
                
                if "Categorical feature" in fill_feat:
                    fill_feat.remove("Categorical feature")
                    fill_feat+=df.select_dtypes(include=object).columns.to_list()
                
                if strategy=="Drop Rows":
                    st.session_state.all_the_process+=f"""
# Drop Rows
df[{fill_feat}] = df[{fill_feat}].dropna(axis=0)
\n """
                    df[fill_feat]=df[fill_feat].dropna(axis=0)
                    st.session_state['df']=df
                    st.success("missing value correspondant row removed")

                elif strategy == "Drop Columns":
                    st.session_state.all_the_process += f"""
# Drop Columns
df[{fill_feat}] = df[{fill_feat}].dropna(axis=1)
\n """
                    df[fill_feat] = df[fill_feat].dropna(axis=1)
                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** have been dropped from the DataFrame.")

                elif strategy=="Median Imputation":
                    st.session_state.all_the_process+= f""""
# Fill with Mean
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
df[{fill_feat}] = num_imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    num_imputer=SimpleImputer(strategy='median')
                    df[fill_feat]=num_imputer.fit_transform(df[fill_feat])   
                    null_cat=df[missing_df_cols].select_dtypes(include=object).columns.to_list()
                    if null_cat:
                        st.session_state.all_the_process +=f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
\n """
                        cat_imputer=SimpleImputer(strategy='most_frequent')
                        df[null_cat]=cat_imputer.fit_transform(df[null_cat])  
                    st.session_state['df'] = df
                    if df.select_dtypes(include=object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean.")       
                elif strategy=="Mode Imputation":
                    st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df[{fill_feat}] = imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[fill_feat] = imputer.fit_transform(df[fill_feat])

                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** has been filled with the Mode.")
        col1,col2,col3=st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="missing_show_df")
        if show_df:
            st.dataframe(df, use_container_width=True)
    
    new_line()
    st.markdown("### Categorical Data Handling",unsafe_allow_html=True)
    new_line()
    with st.expander("Encoding"):
        new_line()
        show_cat=st.checkbox("categorical features",value=False,key='show_cat')
        if show_cat:
            col1,col2=st.columns(2)
            col1.dataframe(df.select_dtypes(include=object),height=250,use_container_width=True)
            if len(df.select_dtypes(include=object).columns.to_list())>1:
                tmp=df.select_dtypes(include=object)
                tmp=tmp.apply(lambda x:x.unique())
                tmp=tmp.to_frame()
                tmp.columns=['Unique Values']
                col2.dataframe(tmp,height=250,use_container_width=True)
        further_analysis = st.checkbox("Further Analysis", value=False, key='further_analysis')
        if further_analysis:

            col1, col2 = st.columns([0.5,1])
            with col1:
                # Each categorical feature has how many unique values as dataframe
                new_line()
                st.markdown("<h6 align='left'> Number of Unique Values", unsafe_allow_html=True)
                unique_values = pd.DataFrame(df.select_dtypes(include=object).nunique())
                unique_values.columns = ['# Unique Values']
                unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                st.dataframe(unique_values, width=200, height=300)

            with col2:
                # Plot for the count of unique values for the categorical features
                new_line()
                st.markdown("<h6 align='center'> Plot for the Count of Unique Values ", unsafe_allow_html=True)
                unique_values = pd.DataFrame(df.select_dtypes(include=object).nunique())
                unique_values.columns = ['# Unique Values']
                unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                unique_values['Feature'] = unique_values.index
                fig = px.bar(unique_values, x='Feature', y='# Unique Values', color='# Unique Values', height=350)
                st.plotly_chart(fig, use_container_width=True)

        
        from sklearn.preprocessing import LabelEncoder

        col1, col2 = st.columns(2)
        with col1:
            enc_feat = st.multiselect("Select Features", df.select_dtypes(include=object).columns.tolist(), key='encoding_feat', help="Select the categorical features to encode.")

        with col2:
            encoding = st.selectbox("Select Encoding", ["Select", "Label Encoding", "Count Frequency Encoding"], key='encoding', help="Select the encoding method.")

        if enc_feat and encoding != "Select":
            new_line()
            new_line()
            col1, col2, col3 = st.columns([1, 0.5, 1])
            
            if col2.button("Apply", key='encoding_apply', use_container_width=True, help="Click to apply encoding."):
                progress_bar()
                new_line()

                if encoding == "Count Frequency Encoding":
                    st.session_state.all_the_process += f"""
        # Count Frequency Encoding
        df[{enc_feat}] = df[{enc_feat}].apply(lambda x: x.map(len(df) / x.value_counts()))
        \n """
                    df[enc_feat] = df[enc_feat].apply(lambda x: x.map(len(df) / x.value_counts()))
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Count Frequency Encoding.")

                elif encoding == "Label Encoding":
                    label_encoders = {}
                    for feature in enc_feat:
                        le = LabelEncoder()
                        df[feature] = le.fit_transform(df[feature])
                        label_encoders[feature] = le  # Store the encoders for inverse transformation if needed
                        
                    st.session_state['df'] = df
                    st.session_state['label_encoders'] = label_encoders  # Save encoders in session state
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Label Encoding.")

        col1, col2, col3 = st.columns([0.15, 1, 0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="cat_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)

        new_line()

    new_line()
    st.markdown("### Scaling",unsafe_allow_html=True)
    new_line()
    with st.expander("Scaling"):
        new_line()
        feat_range = st.checkbox("Further Analysis", value=False, key='feat_range')
        if feat_range:
            new_line()
            st.write("The Ranges for the numeric features:")
            col1, col2, col3 = st.columns([0.05,1, 0.05])
            with col2:
                 st.dataframe(df.describe().T, width=700)
            
            new_line()
        new_line()
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            scale_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features to be scaled.")

        with col2:
            scaling = st.selectbox("Select Scaling", ["Select", "Standard Scaling", "MinMax Scaling"], help="Select the scaling method.")


        if scale_feat and scaling != "Select":       
                new_line()
                col1, col2, col3 = st.columns([1, 0.5, 1])
                
                if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling."):

                    progress_bar()
    
                    # Standard Scaling
                    if scaling == "Standard Scaling":
                        st.session_state.all_the_process += f"""
# Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Standard Scaling.")
    
                    # MinMax Scaling
                    elif scaling == "MinMax Scaling":
                        st.session_state.all_the_process += f"""
# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using MinMax Scaling.")
    

        

        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="scaling_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)
    new_line()


    st.markdown("### 🧬 Data Transformation", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Transformation"):
        new_line()
        


        # Transformation Methods
        trans_methods = st.checkbox("Explain Transformation Methods", key="trans_methods", value=False)
        if trans_methods:
            new_line()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("<h6 align='center'> Log <br> Transformation</h6>", unsafe_allow_html=True)
                st.latex(r'''z = log(x)''')

            with col2:
                st.markdown("<h6 align='center'> Square Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt{x}''')

            with col3:
                st.markdown("<h6 align='center'> Cube Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt[3]{x}''')
            from scipy.stats import skew
            skewness_before = df.apply(skew).sort_values(ascending=False)

            st.markdown("### **Skewness Before Transformation**")
            st.write(skewness_before)

            



        # INPUT
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            trans_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features you want to transform.", key="transformation features")

        with col2:
            trans = st.selectbox("Select Transformation", ["Select", "Log Transformation", "Square Root Transformation", "Cube Root Transformation"],
                                  help="Select the transformation you want to apply.", 
                                  key= "transformation")
        

        if trans_feat and trans != "Select":
            new_line()
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Apply", key='trans_apply',use_container_width=True ,help="Click to apply transformation."):

                progress_bar()

                # new_line()
                # Log Transformation
                if trans == "Log Transformation":
                    st.session_state.all_the_process += f"""
#Log Transformation
df[{trans_feat}] = np.log1p(df[{trans_feat}])
\n """
                    df[trans_feat] = np.log1p(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Log Transformation.")

                # Square Root Transformation
                elif trans == "Square Root Transformation":
                    st.session_state.all_the_process += f"""
#Square Root Transformation
df[{trans_feat}] = np.sqrt(df[{trans_feat}])
\n """
                    df[trans_feat] = np.sqrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Square Root Transformation.")

                # Cube Root Transformation
                elif trans == "Cube Root Transformation":
                    st.session_state.all_the_process += f"""
#Cube Root Transformation
df[{trans_feat}] = np.cbrt(df[{trans_feat}])
\n """
                    df[trans_feat] = np.cbrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Cube Root Transformation.")

                # Exponential Transformation
                
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="trans_show_df", help="Click to show the DataFrame.")
        
        if show_df:
            st.dataframe(df, use_container_width=True)


    # Feature Engineering
    new_line()
    st.markdown("### ⚡ Feature Engineering", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Feature Engineering"):

        # Feature Extraction
        new_line()
        st.markdown("#### Feature Extraction", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        from sklearn.decomposition import PCA
        with col1:
            n_components = st.slider("Number of Principal Components", min_value=1, max_value=min(df.shape[1], 10), value=2, help="Select the number of principal components to extract.")

            col1, col2, col3 = st.columns([1, 0.6, 1])
            new_line()
            if col2.button("Apply PCA"):
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(df.select_dtypes(include=np.number))

                for i in range(n_components):
                    df[f'PC{i+1}'] = principal_components[:, i]

                st.session_state['df'] = df
                st.session_state.all_the_process += f"""
                # Feature Extraction - PCA
                pca = PCA(n_components={n_components})
                principal_components = pca.fit_transform(df.select_dtypes(include=np.number))
                for i in range({n_components}):
                df[f'PC{{i+1}}'] = principal_components[:, i]

                """

                st.success(f"PCA applied with {n_components} components. New features added to the dataset.")




        

        # Feature Selection
        st.divider()
        st.markdown("#### Feature Selection", unsafe_allow_html=True)
        new_line()

        feat_sel = st.multiselect("Select Feature/s", df.columns.tolist(), key='feat_sel', help="Select the Features you want to keep in the dataset")
        new_line()

        if feat_sel:
            col1, col2, col3 = st.columns([1, 0.7, 1])
            if col2.button("Select Features"):
                st.session_state.all_the_process += f"""
# Feature Selection\ndf = df[{feat_sel}]
\n """
                progress_bar()
                new_line()
                df = df[feat_sel]
                st.session_state['df'] = df
                st.success(f"The Features **`{feat_sel}`** have been selected.")
        
        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="feat_eng_show_df", help="Click to show the DataFrame.")
        
        if show_df:
            st.dataframe(df, use_container_width=True)

    new_line()

    st.markdown("### Data Splitting", unsafe_allow_html=True)
    
    
    new_line()
    with st.expander("Data Splitting"):

        new_line()
        train_size, val_size, test_size = 0,0,0
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
            st.session_state['target_variable'] = target
        with col2:
            sets = st.selectbox("Select The Split Sets", ["Select", "Train and Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model. ")
            st.session_state['split_sets'] = sets

        if sets != "Select" and target:
            if sets == "Train, Validation, and Test" :
                new_line()
                col1, col2, col3 = st.columns(3)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    train_size = round(train_size, 2)
                with col2:
                    val_size = st.number_input("Validation Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='val_size')
                    val_size = round(val_size, 2)
                with col3:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='test_size')
                    test_size = round(test_size, 2)

                if float(train_size + val_size + test_size) != 1.0:
                    new_line()
                    st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **validation** + **test** = **{train_size}** + **{val_size}** + **{test_size}** = **{sum([train_size, val_size, test_size])}**" )
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data", use_container_width=True)
                        
                        if split_button:
                            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= {val_size} / (1.0 - {train_size}),random_state=42)
\n """
                            from sklearn.model_selection import train_test_split
                            X_train, X_rem, y_train, y_rem = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= val_size / (1.0 - train_size),random_state=42)
                            st.session_state['X_train'] = X_train
                            st.session_state['X_val'] = X_val
                            st.session_state['X_test'] = X_test
                            st.session_state['y_train'] = y_train
                            st.session_state['y_val'] = y_val
                            st.session_state['y_test'] = y_test

                    
                    col1, col2, col3 = st.columns(3)
                    if split_button:
                        st.success("Data Splitting Done!")
                        with col1:
                            st.write("Train Set")
                            st.write("X Train Shape: ", X_train.shape)
                            st.write("Y Train Shape: ", y_train.shape)

                            train = pd.concat([X_train, y_train], axis=1)
                            train_csv = train.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key='train3')

                        with col2:
                            st.write("Validation Set")
                            st.write("X Validation Shape: ", X_val.shape)
                            st.write("Y Validation Shape: ", y_val.shape)

                            val = pd.concat([X_val, y_val], axis=1)
                            val_csv = val.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Validation Set", val_csv, "validation.csv", key='val3')

                        with col3:
                            st.write("Test Set")
                            st.write("X Test Shape: ", X_test.shape)
                            st.write("Y Test Shape: ", y_test.shape)

                            test = pd.concat([X_test, y_test], axis=1)
                            test_csv = test.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Test Set", test_csv, "test.csv", key='test3')


            elif sets == "Train and Test":

                new_line()
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    train_size = round(train_size, 2)
                with col2:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='val_size')
                    test_size = round(test_size, 2)

                if float(train_size + test_size) != 1.0:
                    new_line()
                    st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **test** = **{train_size}** + **{test_size}** = **{sum([train_size, test_size])}**" )
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data")

                        if split_button:
                            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
\n """
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                            st.session_state['X_train'] = X_train
                            st.session_state['X_test'] = X_test
                            st.session_state['y_train'] = y_train
                            st.session_state['y_test'] = y_test

                    
                    
                    col1, col2 = st.columns(2)
                    if split_button:
                        st.success("Data Splitting Done!")
                        with col1:
                            st.write("Train Set")
                            st.write("X Train Shape: ", X_train.shape)
                            st.write("Y Train Shape: ", y_train.shape)

                            train = pd.concat([X_train, y_train], axis=1)
                            train_csv = train.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Train Set", train_csv, "train.csv", key='train2')

                        with col2:
                            st.write("Test Set")
                            st.write("X test Shape: ", X_test.shape)
                            st.write("Y test Shape: ", y_test.shape)

                            test = pd.concat([X_test, y_test], axis=1)
                            test_csv = test.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Test Set", test_csv, "test.csv", key='test2')


    new_line()
    st.markdown("### building model")
    new_line()
    problem_type=""
    with st.expander("model building"):
        target,problem_type,model="","",""
        col1,col2,col3=st.columns(3)
        with col1:
            target = st.selectbox("Target Variable", [st.session_state['target_variable']] , key='target_ml', help="The target variable is the variable that you want to predict")
            new_line()
        with col2:
            problem_type = st.selectbox("Problem Type", ["Select", "Classification", "Regression"], key='problem_type', help="The problem type is the type of problem that you want to solve")
        with col3:

            if problem_type == "Classification":
                model = st.selectbox("Model", ["Select", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()

            elif problem_type == "Regression":
                model = st.selectbox("Model", ["Linear Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()
        if target != "Select" and problem_type and model:
            
            if problem_type == "Classification":
                 
                if model == "Logistic Regression":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        penalty = st.selectbox("Penalty (Optional)", ["l2", "l1", "none", "elasticnet"], key='penalty')

                    with col2:
                        solver = st.selectbox("Solver (Optional)", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"], key='solver')

                    with col3:
                        C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')

                    
                    col1, col2, col3 = st.columns([1,1,1])
                    if col2.button("Train Model", use_container_width=True):
                        
                        
                        progress_bar()

                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='{penalty}', solver='{solver}', C={C}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(penalty=penalty, solver=solver, C=C, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True,  key='save_model')

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors **Required**", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')

                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')

                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')

                    
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):
                        progress_bar()

                        st.session_state['trained_model_bool'] = True

                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
model.fit(X_train, y_train)
\n """
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Support Vector Machine":
                        
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["rbf", "poly", "linear", "sigmoid", "precomputed"], key='kernel')
    
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=100, value=3, step=1, key='degree')
    
                    with col3:
                        C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')
    
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):

                        progress_bar()
                        st.session_state['trained_model_bool'] = True
    
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Support Vector Machine
from sklearn.svm import SVC
model = SVC(kernel='{kernel}', degree={degree}, C={C}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.svm import SVC
                        model = SVC(kernel=kernel, degree=degree, C=C, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")
    
                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Decision Tree":
                            
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
        
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
        
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                            
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
        
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Random Forest":
                                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                                
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')
    
            if problem_type == "Regression":
                 
                if model == "Linear Regression":
                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fit_intercept = st.selectbox("Fit Intercept (Optional)", [True, False], key='normalize')
            
                    with col2:
                        positive = st.selectbox("Positve (Optional)", [True, False], key='positive')
            
                    with col3:
                        copy_x = st.selectbox("Copy X (Optional)", [True, False], key='copy_x')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept={fit_intercept}, positive={positive}, copy_X={copy_x})
model.fit(X_train, y_train)
\n """
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression(fit_intercept=fit_intercept, positive=positive, copy_X=copy_x)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors (Optional)", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')
            
                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')
            
                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
model.fit(X_train, y_train)
\n """
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')
                if model == "Support Vector Machine":
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["linear", "poly", "rbf", "sigmoid", "precomputed"], key='kernel')
            
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=10, value=3, step=1, key='degree')
            
                    with col3:
                        gamma = st.selectbox("Gamma (Optional)", ["scale", "auto"], key='gamma')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Support Vector Machine
from sklearn.svm import SVR
model = SVR(kernel='{kernel}', degree={degree}, gamma='{gamma}')
model.fit(X_train, y_train)
\n """
                        from sklearn.svm import SVR
                        model = SVR(kernel=kernel, degree=degree, gamma=gamma)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Decision Tree":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
            
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=10, value=2, step=1, key='min_samples_split')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split})
model.fit(X_train, y_train)
\n """
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')
                
                if model == "Random Forest":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=10, value=2, step=1, key='min_samples_split')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split})
model.fit(X_train, y_train)
\n """
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

    
    if st.session_state['trained_model_bool']:
        st.markdown("### 📈 Model Evaluation")

        with st.expander("Model Evaluation"):
            # Load the model
            model = joblib.load('model.pkl')
            model_name = str(model)

            if model_name not in st.session_state['models_with_eval']:
                st.session_state['models_with_eval'][model_name] = []
            
            # Predictions
            if st.session_state["split_sets"] == "Train and Test":
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                st.session_state['y_pred_train'] = y_pred_train
                st.session_state['y_pred_test'] = y_pred_test

            # Select Evaluation Metric
            metric_options = {
                "Classification": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"],
                "Regression": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", 
                            "Root Mean Squared Error (RMSE)", "R2 Score"]
            }
            evaluation_metric = st.multiselect(
                "Select Evaluation Metric", metric_options.get(st.session_state['problem_type'], []), key='evaluation_metric'
            )

            if evaluation_metric:
                metric_functions = {
                    "Accuracy": accuracy_score,
                    "Precision": precision_score,
                    "Recall": recall_score,
                    "F1 Score": f1_score,
                    "AUC Score": roc_auc_score,
                    "Mean Absolute Error (MAE)": mean_absolute_error,
                    "Mean Squared Error (MSE)": mean_squared_error,
                    "Root Mean Squared Error (RMSE)": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                    "R2 Score": r2_score
                }

                results = {}
                for metric in evaluation_metric:
                    if metric not in st.session_state['models_with_eval'][model_name]:
                        st.session_state['models_with_eval'][model_name].append(metric)
                        
                        metric_func = metric_functions.get(metric)
                        if metric_func:
                            train_score = metric_func(y_train, y_pred_train)
                            test_score = metric_func(y_test, y_pred_test)
                            results[metric] = [train_score, test_score]

                # Update session state dataframe
                if results:
                    metrics_df = pd.DataFrame(results, index=['Train', 'Test'])
                    st.session_state['metrics_df'] = metrics_df

            # Display Evaluation Results
            if not st.session_state['metrics_df'].empty:
                st.markdown("### Evaluation Metrics Results")
                st.write(st.session_state['metrics_df'])
                if st.session_state['problem_type'] == "Classification":
                    probabilities = model.predict_proba(X_test)[:,1] 
                    predictions=model.predict(X_test)
                    logistic=ClassificationMetrics(y_test,predictions,probabilities)
                    logistic.display_metrics()
                    logistic.run_all_visualizations()
                elif st.session_state['problem_type'] == "Regression":
                    predictions= model.predict(X_test)
                    Decison=RegressionMetrics(y_test,predictions,X_test)
                    Decison.display_metrics()
                    Decison.run_all_visualizations() 

    st.divider()          
    col1, col2, col3, col4= st.columns(4, gap='small')        

    if col1.button("🎬 Show df", use_container_width=True):
        new_line()
        st.subheader(" 🎬 Show The Dataframe")
        st.write("The dataframe is the dataframe that is used on this application to build the Machine Learning model. You can see the dataframe below 👇")
        new_line()
        st.dataframe(df, use_container_width=True)

    st.session_state.df.to_csv("df.csv", index=False)
    df_file = open("df.csv", "rb")
    df_bytes = df_file.read()
    if col2.download_button("📌 Download df", df_bytes, "df.csv", key='save_df', use_container_width=True):
        st.success("Downloaded Successfully!")

    if col3.button("💻  Code", use_container_width=True):
        new_line()
        st.subheader("💻  The Code")
        st.write("The code below is the code that is used to build the model. It is the code that is generated by the app. You can copy the code and use it in your own project 😉")
        new_line()
        st.code(st.session_state.all_the_process , language='python')

    if col4.button("⛔ Reset", use_container_width=True):
        new_line()
        st.subheader("⛔ Reset")
        st.write("Click the button below to reset the app and start over again")
        new_line()
        st.session_state.reset_1 = True

    if st.session_state.reset_1:
        col1, col2, col3 = st.columns(3)
        if col2.button("⛔ Reset", use_container_width=True, key='reset'):
            st.session_state["reset_1"] = not st.session_state["reset_1"]
        st.write("Reset State:", st.session_state["reset_1"])
        st.session_state.df = None
        st.session_state.clear()
            # st.experimental_rerun()
            






          
    
                    
                







              
