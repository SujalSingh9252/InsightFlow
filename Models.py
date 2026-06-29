import streamlit as st
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
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,classification_report,roc_auc_score,matthews_corrcoef,mean_absolute_error,mean_squared_error,root_mean_squared_error,mean_absolute_percentage_error,explained_variance_score,r2_score,roc_curve,precision_recall_curve



class ClassificationMetrics:
    def __init__(self, y_test, predictions, probabilities):
        """
        Initialize the class with true labels, predicted labels, and predicted probabilities.
        :param y_test: Actual target values.
        :param predictions: Predicted target values.
        :param probabilities: Predicted probabilities (for binary: 1D array, for multiclass: 2D array).
        """
        self.y_test = y_test
        self.predictions = predictions
        self.probabilities = probabilities
        self.num_classes = len(np.unique(y_test))  # Check number of classes

        # Compute metrics
        self.metrics = self.calculate_metrics()
        self.conf_matrix = confusion_matrix(y_test, predictions)
        self.report = classification_report(y_test, predictions, output_dict=True)

    def calculate_metrics(self):
        """Calculate classification metrics and return them as a dictionary."""
        acc = accuracy_score(self.y_test, self.predictions)

        # Set average method based on number of classes
        average_method = "binary" if self.num_classes == 2 else "weighted"

        precision = precision_score(self.y_test, self.predictions, average=average_method)
        recall = recall_score(self.y_test, self.predictions, average=average_method)
        f1 = f1_score(self.y_test, self.predictions, average=average_method)

        # ROC-AUC handling
        if self.num_classes == 2:
            roc_auc = roc_auc_score(self.y_test, self.probabilities)  # Binary case
        else:
            roc_auc = roc_auc_score(self.y_test, self.probabilities, multi_class="ovr", average="weighted")  # Multiclass case

        mcc = matthews_corrcoef(self.y_test, self.predictions)

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Matthews Correlation Coefficient": mcc
        }

    def display_metrics(self):
        """Display metrics as Streamlit components."""
        st.title("Classification Model Evaluation")
        st.metric("Accuracy", f"{self.metrics['Accuracy']:.2f}")

        st.subheader("Precision, Recall, and F1-Score")
        st.write(f"**Precision:** {self.metrics['Precision']:.2f}")
        st.write(f"**Recall:** {self.metrics['Recall']:.2f}")
        st.write(f"**F1-Score:** {self.metrics['F1-Score']:.2f}")

        st.subheader("ROC-AUC and Matthews Correlation Coefficient")
        st.write(f"**ROC-AUC:** {self.metrics['ROC-AUC']:.2f}")
        st.write(f"**Matthews Correlation Coefficient (MCC):** {self.metrics['Matthews Correlation Coefficient']:.2f}")
        st.markdown("---")

    def plot_confusion_matrix(self):
        """Plot the confusion matrix as a heatmap."""
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    def display_classification_report(self):
        """Display the classification report as a DataFrame."""
        st.subheader("Classification Report")
        report_df = pd.DataFrame(self.report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

    def plot_roc_curve(self):
        """Plot the Receiver Operating Characteristic (ROC) curve."""
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()

        if self.num_classes == 2:
            # Binary classification case
            fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
            ax.plot(fpr, tpr, label=f'AUC = {self.metrics["ROC-AUC"]:.2f}')
        else:
            # Multiclass classification case
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(self.y_test == i, self.probabilities[:, i])
                ax.plot(fpr, tpr, label=f'Class {i} AUC = {roc_auc_score(self.y_test == i, self.probabilities[:, i]):.2f}')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

    def plot_precision_recall_curve(self):
        """Plot the Precision-Recall curve."""
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()

        if self.num_classes == 2:
            # Binary classification case
            precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, self.probabilities)
            ax.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
        else:
            # Multiclass classification case
            for i in range(self.num_classes):
                precision_vals, recall_vals, _ = precision_recall_curve(self.y_test == i, self.probabilities[:, i])
                ax.plot(recall_vals, precision_vals, label=f'Class {i}')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')
        st.pyplot(fig)

    def run_all_visualizations(self):
        """Run all visualizations."""
        col1,col2=st.columns(2)
        with col1:

            self.plot_confusion_matrix()
        with col2:
            self.display_classification_report()
        col3,col4=st.columns(2)
        with col3:
            self.plot_roc_curve()
        with col4:
            self.plot_precision_recall_curve()






class RegressionMetrics:
    def __init__(self, y_test, predictions, X_test=None):
        """
        Initialize the class with true values, predicted values, and optionally the feature set.
        :param y_test: Actual target values.
        :param predictions: Predicted target values.
        :param X_test: Feature set used for testing, required for adjusted R².
        """
        self.y_test = y_test
        self.predictions = predictions
        self.X_test = X_test
        self.metrics = self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate regression metrics and return them as a dictionary."""
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        mape = mean_absolute_percentage_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        evs = explained_variance_score(self.y_test, self.predictions)
        
        n = len(self.y_test)
        k = self.X_test.shape[1] if self.X_test is not None else 0
        adjusted_r2_score = 1 - (1 - r2) * (n - 1) / (n - k - 1) if k > 0 else None

        metrics = {
            "Mean Absolute Error (MAE)": mae,
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Absolute Percentage Error (MAPE)": mape,
            "R² Score": r2,
            "Adjusted R² Score": adjusted_r2_score,
            "Explained Variance Score": evs
        }
        return metrics
    
    def display_metrics(self):
        """Display metrics as a Streamlit table."""
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value']).reset_index()
        metrics_df = metrics_df.rename(columns={'index': 'Metric'})
        st.header("📝 Regression Metrics")
        st.table(metrics_df.style.format({"Value": "{:.4f}"}))
        st.markdown("---")
    
    def plot_predicted_vs_actual(self):
        """Plot Predicted vs Actual values."""
        st.header("📊 Predicted vs. Actual Values")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=self.y_test, y=self.predictions, alpha=0.6, ax=ax)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')  # Ideal line
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predicted vs. Actual Values")
        st.pyplot(fig)
    
    def plot_residuals(self):
        """Plot residuals vs predicted values."""
        st.header("📉 Residuals Plot")
        residuals = self.y_test - self.predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=self.predictions, y=residuals, alpha=0.6, ax=ax)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Predicted Values")
        st.pyplot(fig)
    
    def plot_residuals_distribution(self):
        """Plot the distribution of residuals."""
        st.header("📈 Residuals Distribution")
        residuals = self.y_test - self.predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        st.pyplot(fig)
    
    def run_all_visualizations(self):
        """Run all visualizations."""
        self.plot_predicted_vs_actual()
        self.plot_residuals()
        self.plot_residuals_distribution()
