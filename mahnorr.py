import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# App Title
st.title("Linear Regression App")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the file into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(data.head())

    # Select X and Y features
    features = list(data.columns)
    st.write("### Select Features")
    x_feature = st.selectbox("Select X (Independent Variable):", features)
    y_feature = st.selectbox("Select Y (Dependent Variable):", features)

    if st.button("Run Regression"):
        try:
            # Extract X and Y
            X = data[[x_feature]]
            y = data[y_feature]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display results
            st.write("### Model Coefficients")
            st.write(f"Coefficient: {model.coef_[0]}")
            st.write(f"Intercept: {model.intercept_}")

            st.write("### Model Evaluation")
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
