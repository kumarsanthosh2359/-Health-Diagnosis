import streamlit as st
import re
import os
import csv
import base64
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

USER_DB_FILE = r"C:\Users\v santhosh kumar\Desktop\cts\users_db.csv"
IMG_PATH = r"C:\Users\v santhosh kumar\Desktop\cts\sg.jpg"

def initialize_user_file():
    """Ensure users_db.csv exists with correct headers."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["username", "password", "phone_number", "city"])

def username_exists(username):
    """Check if username already exists in users_db.csv."""
    if not os.path.exists(USER_DB_FILE):
        return False
    with open(USER_DB_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["username"].strip().lower() == username.lower():
                return True
    return False

def register_user(username, password, phone_number, city):
    """Register a new user in users_db.csv."""
    username = username.strip()
    password = password.strip()
    phone_number = phone_number.strip()
    city = city.strip()
    with open(USER_DB_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([username, password, phone_number, city])
    st.success(f"Registration Successful! You are now logged in as '{username}'.")
    st.session_state["logged_in"] = True
    st.session_state["username"] = username

def login_user(username, password):
    """Verify login credentials. Return True if valid, else False."""
    if not os.path.exists(USER_DB_FILE):
        return False
    with open(USER_DB_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            stored_username = row["username"].strip().lower()
            stored_password = row["password"].strip()
            if stored_username == username.lower() and stored_password == password:
                return True
    return False

def set_custom_style():
    """Set animated background image with circular motion and adjust text style for readability."""
    if os.path.exists(IMG_PATH):
        with open(IMG_PATH, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        # Animate background using circular movement through keyframes
        style = f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                animation: circularBackground 20s linear infinite;
            }}
            @keyframes circularBackground {{
                0%   {{ background-position: 50% 0%; }}
                25%  {{ background-position: 100% 50%; }}
                50%  {{ background-position: 50% 100%; }}
                75%  {{ background-position: 0% 50%; }}
                100% {{ background-position: 50% 0%; }}
            }}
            .overlay {{
                background-color: rgba(0, 0, 0, 0.5);
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                z-index: -1;
            }}
            body, .stTextInput, .stNumberInput, .stSelectbox, .stCheckbox, .stButton, button, p {{
                font-family: 'Yu Gothic Medium', sans-serif;
                font-size: 30px;
                color: white;
            }}
            .stTextInput > label, .stNumberInput > label {{
                font-size: 30px;
                color: white;
            }}
            </style>
            """
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.markdown("<style>body, p { font-size: 30px; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    """Load training data, train a Decision Tree, return model + metadata."""
    training_file = r"C:\Users\v santhosh kumar\Desktop\cts\Training.csv"
    testing_file = r"C:\Users\v santhosh kumar\Desktop\cts\Testing.csv"

    training = pd.read_csv(training_file)
    testing = pd.read_csv(testing_file)

    cols = training.columns[:-1]
    X = training[cols]
    y = training["prognosis"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.33, random_state=42
    )
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # For symptom grouping usage
    reduced_data = training.groupby(training["prognosis"]).max()

    return clf, le, cols, reduced_data

@st.cache_data
def load_dictionaries():
    """Load severity, description, and precaution data from CSV files."""
    severityDictionary = {}
    description_list = {}
    precautionDictionary = {}

    severity_file = r"C:\Users\v santhosh kumar\Desktop\cts\Symptom-severity.csv"
    description_file = r"C:\Users\v santhosh kumar\Desktop\cts\symptom_Description.csv"
    precaution_file = r"C:\Users\v santhosh kumar\Desktop\cts\symptom_precaution.csv"

    with open(severity_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip header
        for row in csv_reader:
            symptom = row[0].strip()
            severity = int(row[1])
            severityDictionary[symptom] = severity

    with open(description_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0]] = row[1]

    with open(precaution_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

    return severityDictionary, description_list, precautionDictionary

def check_pattern(dis_list, inp):
    """Utility function to match input symptom to known symptoms."""
    inp = inp.replace(" ", "_")
    pattern = re.compile(inp)
    pred_list = [item for item in dis_list if pattern.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp, cols):
    """Second prediction with a new DecisionTree, to confirm or compare results."""
    training_file = r"C:\Users\v santhosh kumar\Desktop\cts\Training.csv"
    df = pd.read_csv(training_file)
    X = df.iloc[:, :-1]
    y = df["prognosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20
    )
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict), dtype=int)
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node, le):
    """Extract disease name from the tree leaf node."""
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return [x.strip() for x in list(disease)]

def calc_condition(exp, days, severityDictionary):
    """Calculate severity condition based on symptom severity & duration."""
    sum_sev = sum(severityDictionary.get(item, 0) for item in exp)
    threshold = (sum_sev * days) / (len(exp) + 1)
    if threshold > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."

def analyze_symptoms(clf, le, cols, reduced_data, disease_input, num_days,
                     severityDictionary, description_list, precautionDictionary):
    """Analyze symptoms and predict disease."""
    symptoms_present = []
    symptoms_exp = []
    tree_ = clf.tree_
    feature_name = [cols[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature]

    def collect_symptoms(node, depth=0):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            val = 1 if name == disease_input else 0
            if val <= tree_.threshold[node]:
                return collect_symptoms(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return collect_symptoms(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node], le)
            red_cols = reduced_data.columns
            row_data = reduced_data.loc[present_disease].values[0]
            symptoms_given = red_cols[row_data.nonzero()]
            return symptoms_given, present_disease

    symptoms_given, present_disease = collect_symptoms(0)
    st.markdown("<div style='text-align: center;'><p>Experiencing any of these symptoms?</p></div>", unsafe_allow_html=True)
    
    for sym in symptoms_given:
        if st.checkbox(sym):
            symptoms_exp.append(sym)
    
    if st.button("Get Diagnosis"):
        with st.spinner("Analyzing your symptoms..."):
            progress_bar = st.progress(0)
            for percent_complete in range(1, 101):
                time.sleep(0.03)
                progress_bar.progress(percent_complete)
            second_prediction = sec_predict(symptoms_exp, cols)
            condition_text = calc_condition(symptoms_exp, num_days, severityDictionary)
            time.sleep(0.5)
        st.info(condition_text)
        if present_disease[0] == second_prediction[0]:
            st.success(f"You may have **{present_disease[0]}**")
            if present_disease[0] in description_list:
                st.write(description_list[present_disease[0]])
        else:
            st.warning(f"You may have **{present_disease[0]}** or **{second_prediction[0]}**")
            if present_disease[0] in description_list:
                st.write(description_list[present_disease[0]])
            st.write("---")
            if second_prediction[0] in description_list:
                st.write(description_list[second_prediction[0]])
        st.markdown("<p style='color: white; font-weight: bold;'>Take the following measures:</p>", unsafe_allow_html=True)
        if present_disease[0] in precautionDictionary:
            for measure in precautionDictionary[present_disease[0]]:
                st.markdown(f"<p style='color: white;'>{measure}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: white;'>No specific measures found.</p>", unsafe_allow_html=True)

def animate_welcome():
    """Animate the welcome message letter by letter."""
    welcome_message = "Welcome to the Health Diagnosis System!"
    animated_text = st.empty()
    display_text = ""
    for char in welcome_message:
        display_text += char
        animated_text.markdown(f"<h1 style='text-align: center;'>{display_text}</h1>", unsafe_allow_html=True)
        time.sleep(0.1)
    time.sleep(0.5)

def animate_login():
    """Animate the login message letter by letter."""
    login_message = "Please Login to Your Account"
    animated_text = st.empty()
    display_text = ""
    for char in login_message:
        display_text += char
        animated_text.markdown(f"<h1 style='text-align: center;'>{display_text}</h1>", unsafe_allow_html=True)
        time.sleep(0.1)
    time.sleep(0.5)

def home():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Welcome to the Health Diagnosis System!</h1>
            <h3>Please select an option below:</h3>
            <p>Register or Login to continue.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """<style>
        .stButton button {
            min-width: 100px;
        }
        </style>""",
        unsafe_allow_html=True
    )
    col1, col2, col3, col4, col5 = st.columns([2, 2, 0.2, 2, 2])
    with col2:
        if st.button("Register"):
            st.session_state["view"] = "register"
    with col4:
        if st.button("Login"):
            st.session_state["view"] = "login"

def main():
    set_custom_style()
    initialize_user_file()
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "view" not in st.session_state:
        st.session_state["view"] = "home"
    if st.session_state["view"] == "home":
        home()
    elif st.session_state["view"] == "register":
        st.markdown("<div style='text-align: center;'><h2>Register</h2></div>", unsafe_allow_html=True)
        new_username = st.text_input("New Username", key="reg_username")
        new_password = st.text_input("New Password", type="password", key="reg_password")
        phone = st.text_input("Phone Number", key="reg_phone")
        city = st.text_input("City", key="reg_city")
        if st.button("Submit Registration"):
            if not new_username or not new_password or not phone or not city:
                st.error("All fields are required!")
            elif username_exists(new_username):
                st.error(f"Username '{new_username}' already taken! Please choose another.")
            else:
                register_user(new_username, new_password, phone, city)
                st.session_state["view"] = "main"
        if st.button("Back to Home"):
            st.session_state["view"] = "home"
    elif st.session_state["view"] == "login":
        st.markdown("<div style='text-align: center;'><h2>Login</h2></div>", unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["view"] = "main"
            else:
                st.error("Invalid username or password!")
        if st.button("Back to Home"):
            st.session_state["view"] = "home"
    elif st.session_state["view"] == "main":
        st.markdown(
            f"<div style='text-align: center;'><h2>Welcome, {st.session_state['username']}!</h2></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='text-align: center;'><p>You can now access the diagnosis functionality below.</p></div>",
            unsafe_allow_html=True
        )
        symptom_input = st.text_input("Enter a symptom you are experiencing (e.g. headache)")
        if symptom_input:
            clf, le, cols, reduced_data = load_and_train()
            severityDictionary, description_list, precautionDictionary = load_dictionaries()
            conf, cnf_dis = check_pattern(cols, symptom_input)
            if conf == 1:
                if len(cnf_dis) > 1:
                    selected_idx = st.selectbox(
                        "Select the one you meant:",
                        range(len(cnf_dis)),
                        format_func=lambda x: cnf_dis[x]
                    )
                    disease_input = cnf_dis[selected_idx]
                else:
                    disease_input = cnf_dis[0]
                num_days = st.number_input("For how many days have you had these symptoms?", min_value=1, step=1)
                analyze_symptoms(
                    clf, le, cols, reduced_data, disease_input, num_days,
                    severityDictionary, description_list, precautionDictionary
                )
            else:
                st.markdown(
                    "<div style='text-align: center; color: red;'>No matching symptom found. Please try another symptom.</div>",
                    unsafe_allow_html=True
                )
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["view"] = "home"
            st.success("You have been logged out.")

if __name__ == "__main__":
    main()
