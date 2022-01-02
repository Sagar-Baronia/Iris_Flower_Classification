import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import datasets

st.write("""
# Iris Flower Prediction Web Application

This application predicts the **Iris flower** type, trained on data fetched from **Sklearn** and Model used for prediction is **Random Forest**.
""")

st.sidebar.header('User Input Parameter')


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X,y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class Label of different flower types and their respective Indices')
st.write(iris.target_names)

st.subheader('Prediction Probability on Input Data')
st.write(prediction_proba)

st.subheader('Final Prediction on Input Data')
st.write(iris.target_names[prediction])
