import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=cols)

# Split data into features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit App UI
st.title('Diabetes Prediction App')

st.write("""
### Enter the following details to check if the person has diabetes:
""")

pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=50, step=1)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, step=0.1)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input('Age', min_value=18, max_value=100, step=1)

# Store input data in a DataFrame
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                          columns=cols[:-1])

# Predict the outcome when the button is clicked
if st.button('Predict Diabetes'):
    prediction = model.predict(input_data)
    
    if prediction == 1:
        st.write("### Result: **Diabetes Detected**")
    else:
        st.write("### Result: **No Diabetes**")

