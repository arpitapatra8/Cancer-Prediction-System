# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:01:15 2022

@author: siddhardhan
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#creating function for prediction
def prediction(user_input):
    
    user_input = np.asarray(user_input)
    user_input = user_input.reshape(1,-1)

    # Predict using Naive Bayes model
    prediction = loaded_model.predict(user_input)
    
    if (prediction[0] == 0):
        return 'Benign'
    else:
        return 'Malignant'
    
    
def main():
    
    #title
    st.title('Cancer Prediction System')
    
    #input data from user
    radius_mean = st.number_input("Enter radius mean", format = "%.2f")
    texture_mean = st.number_input("Enter texture mean", format = "%.2f")
    perimeter_mean = st.number_input("Enter perimeter mean", format = "%.2f")
    area_mean = st.number_input("Enter area mean", format = "%.1f")
    smoothness_mean = st.number_input("Enter smoothness mean", format = "%.5f")
    compactness_mean = st.number_input("Enter compactness mean", format = "%.5f")
    concavity_mean = st.number_input("Enter concavity mean", format = "%.5f")
    concave_points_mean = st.number_input("Enter concave points mean", format = "%.5f")
    symmetry_mean = st.number_input("Enter symmetry mean", format = "%.4f")
    fractal_dimension_mean = st.number_input("Enter fractal dimension mean", format = "%.5f")
    
    #code for predicition
    diagnosis = ''
    
    #result
    
    if st.button('Prediction Result'):
        diagnosis = prediction([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
















