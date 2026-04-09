import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Linear regression model")
uploaded_file = st.file_uploader("Uploder csv file",type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
    columns = df.columns.tolist()
    x_col=st.selectbox("Select Independent column",columns)
    y_col=st.selectbox("Select Dependent column",columns)
    if x_col and y_col:
        x=df[[x_col]]
        y=df[y_col]
        model = LinearRegression()
        model.fit(x,y)
        st.subheader("model cofficients")
        st.write("slope:", model.coef_[0])
        st.write("Intercept:", model.intercept_)
        input_value = st.number_input(f"Enter value for {x_col}",value=0.0)
        if st.button("Predict"):
            prediction = model.predict([[input_value]])
            st.success(f"Predicted {y_col} : {prediction[0]}")


        #Plotting the regression line
        st.subheader("Regression Line")
        plt.scatter(x, y, color='blue', label='Data Points')
        plt.plot(x, model.predict(x), color='red', label='Regression Line')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        st.pyplot(plt)
