import streamlit as st
import re
import os
import csv
import base64
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.tree import DecisionTreeClassifier  # FIXED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- FIXED PATHS FOR IMAGE ----------------
BASE_DIR = os.path.dirname(__file__)
IMG_PATH = os.path.join(BASE_DIR, "assets", "sg.jpg")

# ---------------- YOUR ORIGINAL WINDOWS CSV PATHS ----------------
USER_DB_FILE = r"C:\Users\v santhosh kumar\Desktop\cts\users_db.csv"


def initialize_user_file():
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
    """Background image fix (now works locally)."""
    if os.path.exists(IMG_PATH):
        with open(IMG_PATH, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()

        style = f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)


@st.cache_data
def load_and_train():
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

    reduced_data = training.groupby(training["prognosis"]).max()

    return clf, le, cols, reduced_data


@st.cache_data
def load_dictionaries():
    severityDictionary = {}
    description_list = {}
    precautionDictionary = {}

    severity_file = r"C:\Users\v santhosh kumar\Desktop\cts\Symptom-severity.csv"
    description_file = r"C:\Users\v santhosh kumar\Desktop\cts\symptom_Description.csv"
    precaution_file = r"C:\Users\v santhosh kumar\Desktop\cts\symptom_precaution.csv"

    with open(severity_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        for row in csv_reader:
            severityDictionary[row[0].strip()] = int(row[1])

    with open(description_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0]] = row[1]

    with open(precaution_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

    return severityDictionary, description_list, precautionDictionary


def check_pattern(dis_list, inp):
    inp = inp.replace(" ", "_")
    pattern = re.compile(inp)
    pred_list = [item for item in dis_list if pattern.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp, cols):
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
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return [x.strip() for x in list(disease)]


# ---------------- FIXED _tree.TREE_UNDEFINED USAGE ----------------

def analyze_symptoms(clf, le, cols, reduced_data, disease_input, num_days,
                     severityDictionary, description_list, precautionDictionary):

    symptoms_present = []
    symptoms_exp = []
    tree_ = clf.tree_

    # FIXED: replace undefined check
    feature_name = [cols[i] if i >= 0 else "undefined" for i in tree_.feature]

    def collect_symptoms(node, depth=0):
        if tree_.feature[node] >= 0:  # FIXED
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

    st.write("Experiencing any of these symptoms?")

    for sym in symptoms_given:
        if st.checkbox(sym):
            symptoms_exp.append(sym)

    if st.button("Get Diagnosis"):
        second_prediction = sec_predict(symptoms_exp, cols)
        condition_text = calc_condition(symptoms_exp, num_days, severityDictionary)

        st.info(condition_text)

        if present_disease[0] == second_prediction[0]:
            st.success(f"You may have **{present_disease[0]}**")
        else:
            st.warning(f"You may have **{present_disease[0]}** or **{second_prediction[0]}**")


def calc_condition(exp, days, severityDictionary):
    sum_sev = sum(severityDictionary.get(item, 0) for item in exp)
    threshold = (sum_sev * days) / (len(exp) + 1)
    if threshold > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."


def home():
    st.title("Welcome to Health Diagnosis System")
    col1, col2 = st.columns(2)
    if col1.button("Register"):
        st.session_state["view"] = "register"
    if col2.button("Login"):
        st.session_state["view"] = "login"


def main():
    set_custom_style()
    initialize_user_file()

    if "view" not in st.session_state:
        st.session_state["view"] = "home"

    if st.session_state["view"] == "home":
        home()

    elif st.session_state["view"] == "register":
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password")
        phone = st.text_input("Phone Number")
        city = st.text_input("City")

        if st.button("Submit"):
            if not username or not password or not phone or not city:
                st.error("All fields are required")
            else:
                register_user(username, password, phone, city)
                st.session_state["view"] = "main"

    elif st.session_state["view"] == "login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state["view"] = "main"
                st.session_state["username"] = username
            else:
                st.error("Invalid Username or Password")

    elif st.session_state["view"] == "main":
        st.subheader(f"Welcome {st.session_state['username']}")

        symptom_input = st.text_input("Enter a symptom")

        if symptom_input:
            clf, le, cols, reduced_data = load_and_train()
            severityDictionary, description_list, precautionDictionary = load_dictionaries()
            conf, cnf_dis = check_pattern(cols, symptom_input)

            if conf == 1:
                disease_input = cnf_dis[0]
                num_days = st.number_input("Days", min_value=1, step=1)

                analyze_symptoms(
                    clf, le, cols, reduced_data, disease_input, num_days,
                    severityDictionary, description_list, precautionDictionary
                )

        if st.button("Logout"):
            st.session_state["view"] = "home"


if __name__ == "__main__":
    main()
