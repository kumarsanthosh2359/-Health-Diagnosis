import streamlit as st
import re
import os
import csv
import base64
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

USER_DB_FILE = os.path.join(DATA_DIR, "users_db.csv")
IMG_PATH = os.path.join(BASE_DIR, "sg.jpg")  # Image in main directory

def initialize_user_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["username", "password", "phone_number", "city"])

def username_exists(username):
    if not os.path.exists(USER_DB_FILE):
        return False
    with open(USER_DB_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["username"].strip().lower() == username.lower():
                return True
    return False

def register_user(username, password, phone_number, city):
    with open(USER_DB_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([username, password, phone_number, city])
    st.success(f"Registration Successful! Welcome '{username}'")
    st.session_state["logged_in"] = True
    st.session_state["username"] = username

def login_user(username, password):
    if not os.path.exists(USER_DB_FILE):
        return False

    with open(USER_DB_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["username"].strip() == username and row["password"].strip() == password:
                return True
    return False

def set_custom_style():
    # Check if the image exists at the correct path
    if os.path.exists(IMG_PATH):
        try:
            with open(IMG_PATH, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()

            style = f"""
                <style>
                /* Set background image for the main app */
                .stApp {{
                    background-image: url("data:image/jpg;base64,{encoded_string}");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                    background-position: center;
                }}
                
                /* Make main content area transparent */
                .main {{
                    background-color: transparent !important;
                }}
                
                /* Make all containers transparent */
                .stApp > div {{
                    background-color: transparent !important;
                }}
                
                /* Make blocks semi-transparent for better readability */
                .block-container {{
                    background-color: rgba(255, 255, 255, 0.88) !important;
                    padding: 2rem !important;
                    border-radius: 15px !important;
                    margin: 1rem !important;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
                }}
                
                /* Make sidebars transparent if they exist */
                .css-1d391kg {{
                    background-color: rgba(255, 255, 255, 0.85) !important;
                }}
                
                /* Make headers transparent */
                header {{
                    background-color: transparent !important;
                }}
                
                /* Style for headers */
                h1, h2, h3, h4, h5, h6 {{
                    color: #1a1a1a !important;
                    font-weight: 600 !important;
                }}
                
                /* Style for buttons */
                .stButton > button {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                    color: white !important;
                    border: none !important;
                    padding: 0.5rem 2rem !important;
                    border-radius: 8px !important;
                    font-weight: 500 !important;
                    transition: all 0.3s ease !important;
                }}
                
                .stButton > button:hover {{
                    transform: translateY(-2px) !important;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
                    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
                }}
                
                /* Style for text inputs */
                .stTextInput > div > div > input {{
                    background-color: rgba(255, 255, 255, 0.95) !important;
                    border: 1px solid #ddd !important;
                    border-radius: 8px !important;
                    padding: 0.5rem !important;
                }}
                
                .stTextInput > div > div > input:focus {{
                    border-color: #667eea !important;
                    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
                }}
                
                /* Style for number inputs */
                .stNumberInput > div > div > input {{
                    background-color: rgba(255, 255, 255, 0.95) !important;
                    border: 1px solid #ddd !important;
                    border-radius: 8px !important;
                }}
                
                /* Style for checkboxes */
                .stCheckbox {{
                    background-color: rgba(255, 255, 255, 0.8) !important;
                    padding: 8px !important;
                    border-radius: 8px !important;
                    margin: 5px 0 !important;
                }}
                
                /* Style for success/info/error messages */
                .stAlert, .stSuccess, .stInfo, .stError {{
                    background-color: rgba(255, 255, 255, 0.95) !important;
                    border-radius: 8px !important;
                    padding: 1rem !important;
                    margin: 0.5rem 0 !important;
                }}
                
                /* Style for markdown content */
                .stMarkdown {{
                    background-color: transparent !important;
                }}
                
                /* Style for expanders */
                .streamlit-expanderHeader {{
                    background-color: rgba(255, 255, 255, 0.9) !important;
                    border-radius: 8px !important;
                }}
                
                /* Style for select boxes */
                .stSelectbox > div > div {{
                    background-color: rgba(255, 255, 255, 0.95) !important;
                    border-radius: 8px !important;
                }}
                </style>
            """
            st.markdown(style, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load background image: {e}")
    else:
        st.warning(f"Background image not found at: {IMG_PATH}")

@st.cache_data
def load_and_train():
    training_file = os.path.join(DATA_DIR, "Training.csv")
    
    # Check if training file exists
    if not os.path.exists(training_file):
        st.error(f"Training file not found at: {training_file}")
        return None, None, None, None

    training = pd.read_csv(training_file)

    cols = training.columns[:-1]
    X = training[cols]
    y = training["prognosis"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    reduced_data = training.groupby("prognosis").max()

    return clf, le, cols, reduced_data

@st.cache_data
def load_dictionaries():
    severityDictionary = {}
    description_list = {}
    precautionDictionary = {}

    severity_file = os.path.join(DATA_DIR, "Symptom-severity.csv")
    description_file = os.path.join(DATA_DIR, "symptom_Description.csv")
    precaution_file = os.path.join(DATA_DIR, "symptom_precaution.csv")

    # Load severity dictionary
    if os.path.exists(severity_file):
        with open(severity_file, encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                severityDictionary[row[0]] = int(row[1])

    # Load description dictionary
    if os.path.exists(description_file):
        with open(description_file, encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                description_list[row[0]] = row[1]

    # Load precaution dictionary
    if os.path.exists(precaution_file):
        with open(precaution_file, encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                precautionDictionary[row[0]] = row[1:]

    return severityDictionary, description_list, precautionDictionary

def check_pattern(dis_list, inp):
    inp = inp.replace(" ", "_")
    pattern = re.compile(inp, re.IGNORECASE)
    pred_list = [item for item in dis_list if pattern.search(item)]
    return (1, pred_list) if pred_list else (0, [])

def sec_predict(symptoms_exp, cols):
    training_file = os.path.join(DATA_DIR, "Training.csv")
    
    if not os.path.exists(training_file):
        return ["No training data available"]
        
    df = pd.read_csv(training_file)
    X = df.iloc[:, :-1]
    y = df["prognosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))

    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    return model.predict([input_vector])

def calc_condition(exp, days, severityDictionary):
    if not exp:
        return "No symptoms selected. Please select at least one symptom."
    total = sum(severityDictionary.get(i, 0) for i in exp)
    score = (total * days) / (len(exp) + 1)
    
    if score > 13:
        return "⚠️ **You should see a doctor immediately.** Your symptoms indicate a serious condition."
    elif score > 8:
        return "🏥 **You should consult a doctor soon.** Your symptoms need medical attention."
    else:
        return "💊 **Take precautions, but it's not severe.** Rest and monitor your symptoms."

def analyze_symptoms(clf, le, cols, reduced_data, disease_input, days, sev, desc, prec):
    if clf is None or le is None or cols is None or reduced_data is None:
        st.error("Model not loaded properly. Please check your data files.")
        return

    try:
        tree_ = clf.tree_
        feature_name = [cols[i] if i >= 0 else "undefined" for i in tree_.feature]

        symptoms_taken = []
        symptoms_exp = []

        def walk_tree(node):
            if tree_.feature[node] >= 0:
                name = feature_name[node]
                val = 1 if name == disease_input else 0
                if val <= tree_.threshold[node]:
                    return walk_tree(tree_.children_left[node])
                else:
                    symptoms_taken.append(name)
                    return walk_tree(tree_.children_right[node])
            else:
                node_values = tree_.value[node][0]
                final_idx = np.argmax(node_values)
                final_disease = le.inverse_transform([final_idx])[0]
                row = reduced_data.loc[final_disease]
                symptoms_list = list(row[row == 1].index)
                return symptoms_list, [final_disease]

        symptoms_list, predicted = walk_tree(0)

        st.markdown("### 🩺 Select your symptoms:")
        st.markdown("---")
        
        # Create columns for better layout
        cols_per_row = 2
        symptom_cols = st.columns(cols_per_row)
        
        for idx, sym in enumerate(symptoms_list):
            with symptom_cols[idx % cols_per_row]:
                if st.checkbox(sym, key=f"symptom_{sym}"):
                    symptoms_exp.append(sym)

        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔍 Diagnose", use_container_width=True):
                if not symptoms_exp:
                    st.warning("Please select at least one symptom to diagnose.")
                    return
                    
                with st.spinner("Analyzing your symptoms..."):
                    time.sleep(1)  # Simulate processing
                    
                    statement = calc_condition(symptoms_exp, days, sev)
                    
                    st.markdown("---")
                    st.info(statement)
                    
                    st.success(f"### 🏥 Possible Disease: **{predicted[0]}**")
                    
                    st.markdown("### 📋 Description:")
                    st.write(desc.get(predicted[0], "No description available."))
                    
                    st.markdown("### 🛡️ Precautions:")
                    precautions = prec.get(predicted[0], [])
                    if precautions:
                        for i, p in enumerate(precautions, 1):
                            if p:  # Check if precaution is not empty
                                st.markdown(f"{i}. {p}")
                    else:
                        st.write("No specific precautions available.")
                    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

def main():
    # Set page config
    st.set_page_config(
        page_title="Health Diagnosis System",
        page_icon="🏥",
        layout="wide"
    )
    
    set_custom_style()
    initialize_user_file()

    if "view" not in st.session_state:
        st.session_state["view"] = "home"

    if st.session_state["view"] == "home":
        st.title("🏥 Health Diagnosis System")
        st.markdown("### Your Personal Health Assistant")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Welcome to Health Diagnosis System")
            st.markdown("Get instant health insights based on your symptoms")
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📝 Register", use_container_width=True):
                    st.session_state["view"] = "register"
            with col_btn2:
                if st.button("🔑 Login", use_container_width=True):
                    st.session_state["view"] = "login"

    elif st.session_state["view"] == "register":
        st.title("📝 Registration")
        st.markdown("Create your account to get started")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            phone = st.text_input("📱 Phone Number", placeholder="Enter your phone number")
            city = st.text_input("🏙️ City", placeholder="Enter your city")
            
            st.markdown("---")
            
            if st.button("✅ Register", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all required fields")
                elif username_exists(username):
                    st.error("Username already exists")
                else:
                    register_user(username, password, phone, city)
                    st.session_state["view"] = "main"
                    st.rerun()
            
            if st.button("← Back to Home", use_container_width=True):
                st.session_state["view"] = "home"
                st.rerun()

    elif st.session_state["view"] == "login":
        st.title("🔑 Login")
        st.markdown("Access your health dashboard")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            st.markdown("---")
            
            if st.button("🔓 Login", use_container_width=True):
                if login_user(username, password):
                    st.session_state["username"] = username
                    st.session_state["view"] = "main"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            if st.button("← Back to Home", use_container_width=True):
                st.session_state["view"] = "home"
                st.rerun()

    elif st.session_state["view"] == "main":
        st.title(f"👋 Welcome, {st.session_state['username']}!")
        st.markdown("### Symptom Checker")
        st.markdown("---")
        
        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symptom_input = st.text_input(
                "🔍 Enter a symptom:",
                placeholder="e.g., headache, fever, cough...",
                help="Start typing a symptom and we'll suggest related ones"
            )
            
            if symptom_input:
                with st.spinner("Loading disease models..."):
                    clf, le, cols, reduced_data = load_and_train()
                    sev, desc, prec = load_dictionaries()
                
                if clf is not None:
                    conf, matches = check_pattern(cols, symptom_input)
                    
                    if conf:
                        disease_input = matches[0]
                        st.success(f"✅ Found related symptom: **{disease_input}**")
                        days = st.number_input(
                            "📅 How many days have you been experiencing this?",
                            min_value=1,
                            max_value=30,
                            value=1,
                            step=1
                        )
                        
                        analyze_symptoms(clf, le, cols, reduced_data, disease_input, days, sev, desc, prec)
                    else:
                        st.error("❌ Symptom not found in our database. Please try a different symptom.")
                else:
                    st.error("Unable to load the diagnosis system. Please check your data files.")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("---")
            st.info("""
            **How to use:**
            1. Enter a symptom you're experiencing
            2. Select related symptoms from the list
            3. Specify how many days you've had them
            4. Click 'Diagnose' for analysis
            
            ⚠️ **Note:** This is not a substitute for professional medical advice.
            """)
            
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state["view"] = "home"
                st.rerun()
        
        # Add footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: gray;'>© 2024 Health Diagnosis System | For informational purposes only</p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
