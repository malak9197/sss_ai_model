import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report


real_names_map = {
    "Person1": "person1", 
    "Person2": "person2", 
    "Person3": "person3", 
    "Person4": "person4"
}

DISPLAY_NAMES = [real_names_map[p] for p in ["Person1", "Person2", "Person3", "Person4"]]


st.set_page_config(page_title="SSS Performance Dashboard", layout="wide")

st.title("🛡️ SSS: Smart Security & Safety Analysis")
st.markdown(f"**Team Leader:** Malak | **Project:** Smart Security System")
st.markdown("---")


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Overall Accuracy", "100%", help="Model performance on 149 test samples")
with col2:
    st.metric("Avg Confidence", "99.23%", delta="High Precision")
with col3:
    st.metric("Inference Speed", "0.01 ms", delta_color="inverse")

st.markdown("---")


col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📈 Confusion Matrix (Real Names)")
   
    cm = np.array([[39, 0, 0, 0], [0, 35, 0, 0], [0, 0, 38, 0], [0, 0, 0, 37]])
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES, ax=ax)
    plt.ylabel('Actual Person')
    plt.xlabel('Predicted Person')
    st.pyplot(fig)

with col_right:
    st.subheader("🎯 Individual Class Performance")
    performance_df = pd.DataFrame({
        "Person": DISPLAY_NAMES,
        "Correct": [39, 35, 38, 37],
        "Total": [39, 35, 38, 37],
        "Accuracy": ["100%", "100%", "100%", "100%"]
    })
    st.table(performance_df)

st.markdown("---")


st.subheader("📋 System Access Logs (Latest Activity)")


LOG_FILE = '/mnt/d/malak/sss_ai/face-recognition-6/test/analysis_results.csv'

if os.path.exists(LOG_FILE):
    df_logs = pd.read_csv(LOG_FILE)
    
    st.dataframe(df_logs.iloc[::-1].head(40), width='stretch')
else:
    st.info("No logs found. Please run the Scanner and ensure 'access_logs.csv' is in the same folder.")


st.markdown("---")
st.subheader("🛡️ Prediction Confidence Analysis")

if os.path.exists(LOG_FILE):
   
    if 'Confidence' in df_logs.columns:
        
        
        df_logs['Conf_Value'] = df_logs['Confidence'].astype(str).str.replace('%', '').astype(float) / 100
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.write("Latest Predictions Analysis:")
            
            if 'User' in df_logs.columns:
                display_cols = ['User', 'Confidence']
            elif 'filename' in df_logs.columns:
                display_cols = ['filename', 'Confidence']
            else:
                display_cols = ['Confidence']
            
            
            st.table(df_logs[display_cols].iloc[::-1].head(10))
            
        with col_chart:
            st.write("Confidence Distribution Histogram")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
           
            sns.histplot(df_logs['Conf_Value'], bins=10, kde=True, color="green", ax=ax2)
            ax2.set_xlim(0, 1.05)
            ax2.set_xlabel("Confidence Level (Probability)")
            ax2.set_ylabel("Number of Samples")
            st.pyplot(fig2)