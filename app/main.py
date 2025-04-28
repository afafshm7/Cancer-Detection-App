import pickle as pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv('D:/Anaconda ML/Cancer Detection App/Data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    sidebar_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave Points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal Dimension (mean)", "fractal_dimension_mean"),
    
    ("Radius SE", "radius_se"),
    ("Texture SE", "texture_se"),
    ("Perimeter SE", "perimeter_se"),
    ("Area SE", "area_se"),
    ("Smoothness SE", "smoothness_se"),
    ("Compactness SE", "compactness_se"),
    ("Concavity SE", "concavity_se"),
    ("Concave Points SE", "concave points_se"),
    ("Symmetry SE", "symmetry_se"),
    ("Fractal Dimension SE", "fractal_dimension_se"),

    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave Points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal Dimension (worst)", "fractal_dimension_worst")
    ]    

    input_dict = {}

    for label, id in sidebar_labels:
        input_dict[id] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[id].max()),
            value=float(data[id].mean()),
            key=f"{label}, {id}"
        ) 
    return input_dict
        
def get_sclaed_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_sclaed_values(input_data)

    categories = ['ٌRadius','Texture','Perimeter',
              'Area', 'Smoothness', 'Compactness',
              'Concavity', 'Concave Points',
              'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'],
            input_data['perimeter_mean'], input_data['area_mean'],
            input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'],
            input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'],
            input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'],
            input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'],
            input_data['perimeter_worst'], input_data['area_worst'],
            input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'],
            input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=False
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("Model/model.pkl", "rb"))
    scaler = pickle.load(open("Model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled = scaler.transform(input_array)
    predction = model.predict(input_array_scaled)
    
    st.subheader("Cell Cluster Prediction")
    st.write("The Cell is:")

    if predction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

    st.write(f"Probability of being benign: {model.predict_proba(input_array_scaled)[0][0]}")
    st.write(f"Probability of being Malicious: {model.predict_proba(input_array_scaled)[0][1]}")

    st.write("This app can assist medical profesinals in making a diagnosis, however it should NOT replace a professinal one. ")
def main():
    st.set_page_config(
        page_title=("Breast Cancer Predictor"),
        page_icon="female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()
#    st.write(input_data)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please conncet this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app uses a machine learning model to decide whether a breast mass is Banign or Malicious based on the mesurmeents it receives from your cytosis lab. You can also update the measurements by hand using the slides in the sidebare")
        
        col1, col2 = st.columns([4,1])
        
        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
        with col2:
            add_predictions(input_data)

if __name__ == '__main__':
    main()