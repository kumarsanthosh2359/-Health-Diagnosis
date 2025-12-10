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

# ======================================================
# FIXED: RELATIVE PATHS FOR STREAMLIT CLOUD
# ======================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

USER_DB_FILE = os.path.join(DATA_DIR, "users_db.csv")
IMG_PATH = os.path.join(ASSETS_DIR, "sg.jpg")

# ======================================================
# USER FUNCTIONS
# ======================================================
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


# ======================================================
# ANIMATED BACKGROUND
# ======================================================
def set_custom_style():
    if os.path.exists(IMG_PATH):
        with open(IMG_PATH, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;

                /* 🔥 ANIMATION */
                animation: moveBg 25s ease-in-out infinite;
            }}

            @keyframes moveBg {{
                0%   {{ background-position: 50% 0%; }}
                25%  {{ background-position: 100% 50%; }}
                50%  {{ background-position: 50% 100%; }}
                75%  {{ background-position: 0% 50%; }}
                100% {{ background-position: 50% 0%; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


# ======================================================
# LOAD & TRAIN MODEL
# ======================================================
@st.cache_data
def load_and_train():
    training_file = os.path.join(DATA_DIR, "Training.csv")
    testing_file = os.path.join(DATA_DIR, "Testing.csv")

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


# ======================================================
# LOAD DICTIONARIES
# ======================================================
@st.cache_data
def load_dictionaries():
    severityDictionary = {}
    description_list = {}
    precautionDictionary = {}

    severity_file = os.path.join(DATA_DIR, "Symptom-severity.csv")
    description_file = os.path.join(DATA_DIR, "symptom_Description.csv")
    precaution_file = os.path.join(DATA_DIR, "symptom_precaution.csv")

    with open(severity_file, encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            severityDictionary[row[0]] = int(row[1])

    with open(description_file, encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            description_list[row[0]] = row[1]

    with open(precaution_file, encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            precautionDictionary[row[0]] = row[1:]

    return severityDictionary, description_list, precautionDictionary


# ======================================================
# SUPPORT FUNCTIONS
# ======================================================
def check_pattern(dis_list, inp):
    inp = inp.replace(" ", "_")
    pattern = re.compile(inp)
    pred_list = [item for item in dis_list if pattern.search(item)]
    return (1, pred_list) if pred_list else (0, [])


def sec_predict(symptoms_exp, cols):
    df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
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
    total = sum(severityDictionary.get(i, 0) for i in exp)
    score = (total * days) / (len(exp) + 1)
    return "You should see a doctor." if score > 13 else "Take precautions."


# ======================================================
# ANALYZE SYMPTOMS
# ======================================================
def analyze_symptoms(clf, le, cols, reduced_data, disease_input, days, sev, desc, prec):

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

    st.subheader("Select your symptoms:")
    for sym in symptoms_list:
        if st.checkbox(sym):
            symptoms_exp.append(sym)

    if st.button("Diagnose"):
        statement = calc_condition(symptoms_exp, days, sev)

        st.info(statement)

        st.success(f"Possible Disease: **{predicted[0]}**")

        st.write("### Description:")
        st.write(desc.get(predicted[0], "No description available."))

        st.write("### Precautions:")
        for p in prec.get(predicted[0], []):
            st.write("-", p)


# ======================================================
# MAIN UI
# ======================================================
def main():
    set_custom_style()
    initialize_user_file()

    if "view" not in st.session_state:
        st.session_state["view"] = "home"

    if st.session_state["view"] == "home":
        st.title("Health Diagnosis System")

        col1, col2 = st.columns(2)
        if col1.button("Register"):
            st.session_state["view"] = "register"
        if col2.button("Login"):
            st.session_state["view"] = "login"

    elif st.session_state["view"] == "register":
        st.header("Register")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        phone = st.text_input("Phone Number")
        city = st.text_input("City")

        if st.button("Submit"):
            if username_exists(username):
                st.error("Username already exists")
            else:
                register_user(username, password, phone, city)
                st.session_state["view"] = "main"

    elif st.session_state["view"] == "login":
        st.header("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state["username"] = username
                st.session_state["view"] = "main"
            else:
                st.error("Invalid credentials")

    elif st.session_state["view"] == "main":
        st.header(f"Welcome {st.session_state['username']}")

        symptom_input = st.text_input("Enter a symptom:")

        if symptom_input:
            clf, le, cols, reduced_data = load_and_train()
            sev, desc, prec = load_dictionaries()

            conf, matches = check_pattern(cols, symptom_input)

            if conf:
                disease_input = matches[0]
                days = st.number_input("How many days?", min_value=1, step=1)

                analyze_symptoms(clf, le, cols, reduced_data, disease_input, days, sev, desc, prec)
            else:
                st.error("Symptom not found.")

        if st.button("Logout"):
            st.session_state["view"] = "home"


if __name__ == "__main__":
    main()
