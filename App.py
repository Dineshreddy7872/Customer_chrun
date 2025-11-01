import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests

# Page config
st.set_page_config(page_title="Customer Churn Prediction",
                   page_icon="📊",
                   layout="centered",
                   initial_sidebar_state="expanded")

# Load animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_t24tpvcu.json")

# Load the trained model
model = joblib.load(r"C:\Users\user\OneDrive\Desktop\My Projects\Customer_churn_project\final_gb_classifier.pkl")

# Preprocess function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })
    return df

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=100)
    st.title("📊 Churn Predictor")
    st.markdown("### Input Customer Details Below 👇")

# Main Header
st.markdown("<h1 style='text-align:center; color:#00BFFF;'>Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")
st_lottie(lottie_animation, height=200, key="churn_anim")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    senior_citizen = st.radio("Senior Citizen", [0, 1])
    partner = st.radio("Partner", [0, 1])
    dependents = st.radio("Dependents", [0, 1])
    phone_service = st.radio("Phone Service", [0, 1])
    multiple_lines = st.radio("Multiple Lines", [0, 1])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

with col2:
    online_security = st.selectbox("Online Security", [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet'][x])
    online_backup = st.selectbox("Online Backup", [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet'][x])
    device_protection = st.selectbox("Device Protection", [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet'][x])
    tech_support = st.selectbox("Tech Support", [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet'][x])
    streaming_tv = st.radio("Streaming TV", [0, 1])
    streaming_movies = st.radio("Streaming Movies", [0, 1])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.radio("Paperless Billing", [0, 1])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

monthly_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges ($)", value=0.0)
tenure_group = st.number_input("Tenure Group (Months)", value=1, min_value=0, max_value=100)

# Prediction button
st.write("---")
predict_btn = st.button("🔮 Predict Customer Churn", use_container_width=True)

if predict_btn:
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }

    processed_data = preprocess_input(user_data)
    prediction = model.predict(processed_data)

    st.success("✅ Prediction Complete!")

    if prediction[0] == 1:
        st.error("🚨 The customer is **likely to churn!** Take retention actions immediately.")
    else:
        st.balloons()
        st.info("🎉 The customer is **likely to stay.**")

st.write("---")
st.caption("Built with ❤️ using Streamlit | Model: Gradient Boosting Classifier")
