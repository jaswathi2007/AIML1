import streamlit as st
import pandas as pd
import joblib

# Load saved model and features
model = joblib.load("titanic_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 0.0, 80.0, 28.0)
    embarked = st.selectbox("Embarkation Port", ["Southampton", "Cherbourg", "Queenstown"])

with col2:
    fare = st.number_input("Fare Paid ($)", value=15.0, min_value=0.0)
    sibsp = st.number_input("Siblings / Spouses aboard", value=0, min_value=0)
    parch = st.number_input("Parents / Children aboard", value=0, min_value=0)

if st.button("Predict Survival", type="primary"):
    # Replicate notebook feature engineering
    is_alone = 1 if (sibsp == 0 and parch == 0) else 0
    is_male = 1 if sex == "male" else 0
    is_adult_male = 1 if (is_male and age >= 16) else 0

    input_data = {
        'pclass': pclass,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'adult_male': is_adult_male,
        'alone': is_alone,
        'sex_male': sex == "male",
        'embarked_Q': embarked == "Queenstown",
        'embarked_S': embarked == "Southampton",
        'who_man': (sex == "male" and age >= 16),
        'who_woman': (sex == "female" and age >= 16)
    }

    # Filter out 'embark_town' to match the actual input features
    features = [col for col in model_features if col != 'embark_town']
    input_df = pd.DataFrame([input_data])[features]

    prediction = model.predict(input_df)[0]
    prob_survived = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"**Survived!** Survival Chance: **{prob_survived * 100:.1f}%**")
    else:
        st.error(f"**Did not survive.** Survival Chance: **{prob_survived * 100:.1f}%**")