import streamlit as st
import pandas as pd
import base64
from pathlib import Path
import time
from mian import prediction_function

# Function to add background image and custom CSS

def add_bg_and_styling(image_file):
    try:
        with open(image_file, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()  # Convert bytes to string
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .input-container {{
                background-color: #333333;
                padding: 30px;
                border-radius: 15px;
                color: #f5f5f5;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                width: 80%;
                margin: 0 auto;
                margin-top: 20px;
            }}
            h1, h2 {{
                color: darkkhaki;
            }}
            label {{
                color: burlywood !important;
                font-weight: bold;
            }}
            input, select, .stSelectbox:first-of-type > div[data-baseweb="select"] > div  {{
                background-color: #E8E8E8 !important;
                color: #000000 !important;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }}
            .stButton > button , .stFormSubmitButton > button{{
                background-color: burlywood !important;
                color: brown;
                border-radius: 5px;
                border: none;
                padding: 10px 24px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}
            .stButton > button:hover, .stFormSubmitButton > button:hover{{
                background-color: sandybrown !important;
            }}
            .credit-card-loader {{
                display: block;
                margin: 0 auto;
                margin-top: 20px;
                width: 100px;
                height: 60px;
                border-radius: 10px;
                background: linear-gradient(135deg, #3a3a3a 50%, #8b8b8b 50%);
                position: relative;
                animation: card-spin 1.5s infinite ease-in-out;
            }}
            @keyframes card-spin {{
                0% {{
                    transform: rotateY(0deg);
                }}
                50% {{
                    transform: rotateY(180deg);
                }}
                100% {{
                    transform: rotateY(360deg);
                }}
            }}
            .result-box {{
                padding: 20px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                margin-top: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Background image not found. Please check the path.")

# Function to simulate a loading effect
import time
import random

def effect():
    x = random.randint(1,10)  # Generate a random integer between 2 and 8
    time.sleep(x)  # Simulate a delay for the loader effect
    return

# Set page title and layout
st.set_page_config(page_title="Credit Card Prediction App", layout="centered", initial_sidebar_state="collapsed")

# Add background image and styling
image_path = "image_3.jpg"  # Update this path to your image file
if Path(image_path).exists():
    add_bg_and_styling(image_path)
else:
    st.warning(f"Background image not found at {image_path}. Please check the path.")

# Page title
st.markdown("<h1 style='text-align: center;'>Credit Card Eligibility Predictor</h1>", unsafe_allow_html=True)

# Create a container with better contrast for input fields
with st.container():
    st.markdown("<h2>Fill in your details to check your eligibility for a credit card!</h2>", unsafe_allow_html=True)

    # Input form inside the input-container div
    with st.form("input_form", clear_on_submit=True):
        # Dropdowns and inputs
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ['', 'M', 'F'])
            has_car = st.selectbox("Has a car", ['', 'YES', 'NO'])
            has_property = st.selectbox("Has a property", ['', 'YES', 'NO'])
            income = st.number_input("Income (yearly (in rupees))", min_value=0, value=0)
            employment_status = st.selectbox("Employment status", ['', 'Commercial associate', 'Pensioner', 'State servant', 'Student', 'Working'])
            education_level = st.selectbox("Education level", ['', 'Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree'])
            marital_status = st.selectbox("Marital status", ['', 'Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow'])
            dwelling = st.selectbox("Dwelling", ['', 'Co-op apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents'])

        with col2:
            age = st.number_input("Age", min_value=0, value=0)
            employment_length = st.number_input("Employment length (in years)", min_value=0.0, value=0.0)
            has_mobile = st.selectbox("Has a mobile phone", ['', 'YES', 'NO'])
            has_work_phone = st.selectbox("Has a work phone", ['', 'YES', 'NO'])
            has_phone = st.selectbox("Has a phone", ['', 'YES', 'NO'])
            has_email = st.selectbox("Has an email", ['', 'YES', 'NO'])

            # Updated list with 50 most common jobs in India
            job_titles = [
    '', 'Accountant', 'Administrative Assistant', 'Architect', 'Auditor', 'Bank Clerk', 'Bank Manager', 
    'Barista', 'Bartender', 'Business Analyst', 'Business Development Executive', 'Cabin Crew', 'Carpenter', 
    'Cashier', 'Chartered Accountant', 'Chef', 'Civil Engineer', 'Cleaner', 'Construction Manager', 
    'Consultant', 'Content Writer', 'Customer Service Executive', 'Data Analyst', 'Data Scientist', 
    'Database Administrator', 'Delivery Driver', 'Dentist', 'Doctor', 'Driver', 'Electrician', 'Electrical Engineer', 
    'Executive Assistant', 'Event Planner', 'Fashion Designer', 'Financial Analyst', 'Graphic Designer', 
    'Hair Stylist', 'HR Manager', 'Insurance Agent', 'IT Support Specialist', 'Journalist', 'Laboratory Technician', 
    'Lawyer', 'Logistics Manager', 'Machine Operator', 'Maintenance Worker', 'Manager', 'Marketing Executive', 
    'Mechanical Engineer', 'Medical Assistant', 'Nurse', 'Operations Manager', 'Pharmaceutical Sales', 
    'Pharmacist', 'Photographer', 'Physical Therapist', 'Pilot', 'Plumber', 'Policy Advisor', 'Police Officer', 
    'Production Supervisor', 'Project Manager', 'Quality Analyst', 'Real Estate Agent', 'Receptionist', 'Recruiter', 
    'Research Scientist', 'Retail Manager', 'Sales Executive', 'Sales Manager', 'Security Guard', 'SEO Specialist', 
    'Social Media Manager', 'Software Engineer', 'Software Tester', 'Sound Engineer', 'Speech Therapist', 
    'Store Manager', 'System Administrator', 'Teacher', 'Technical Support Specialist', 'Technical Writer', 
    'Tour Guide', 'Training Coordinator', 'Translator', 'Travel Agent', 'Truck Driver', 'UX Designer', 
    'Veterinarian', 'Video Editor', 'Waiter/Waitress', 'Web Developer', 'Welder', 'Writer', 'Yoga Instructor', 
    'Zoologist','others'
]
            job_title = st.selectbox("Job title", job_titles)
            
            family_member_count = st.number_input("Family member count", min_value=0, value=0)

        # Submit button
        submitted = st.form_submit_button("Predict")

    # Placeholder for loader and result
    loader_placeholder = st.empty()
    result_placeholder = st.empty()

    # Validation: Check if any required field is left empty
    if submitted:
        if (not gender or not has_car or not has_property or income == 0 or not employment_status or 
            not education_level or not marital_status or not dwelling or age == 0 or employment_length == 0.0 or 
            not has_mobile or not has_work_phone or not has_phone or not has_email or not job_title or family_member_count == 0):
            result_placeholder.error("Please fill in all details before predicting.")
        else:
            # Display loader in the placeholder
            loader_placeholder.markdown('<div class="credit-card-loader"></div>', unsafe_allow_html=True)
            
            

            # Simulate prediction process
            ans = ""
            user_data = [gender, has_car, has_property, 0, income, employment_status, education_level, marital_status, dwelling, age, employment_length, has_mobile, has_work_phone, has_phone, has_email, job_title, family_member_count]
            
            result = prediction_function(user_data)
            effect()

            # Remove loader
            loader_placeholder.empty()

            # Conditional formatting based on the prediction
            if result == 1:
                ans = "<div class='result-box' style='color: green; background-color: #e6ffe6;'>Your credit card request will be approved!</div>"
            elif result == -1:
                ans = "<div class='result-box' style='color: orange; background-color: #e6ffe6;'>Unfortunately, you are ineligible for a credit card as your age is below 18.</div>"
            else:
                ans = "<div class='result-box' style='color: red; background-color: #ffe6e6;'>Sorry, your credit card application will be rejected.</div>"

            # Display result in the result_placeholder
            result_placeholder.markdown(f"{ans}", unsafe_allow_html=True)


# Add footer
st.markdown("<hr><footer style='text-align: center; color: #888;'>Powered by Streamlit | Created by Tushar M Awale</footer>", unsafe_allow_html=True)