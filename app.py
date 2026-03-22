import streamlit as st
import re
import os
import csv
import base64
import pandas as pd
import numpy as np
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")

USER_DB_FILE = os.path.join(DATA_DIR, "users_db.csv")

# Possible background images (add more names if needed)
POSSIBLE_BG_IMAGES = ["sg.jpg", "o.jpg", "pl.jpg"]


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
    st.success(f"Registration Successful! Welcome **{username}**")
    st.session_state["logged_in"] = True
    st.session_state["username"] = username
    st.session_state["view"] = "main"


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
    """Improved background with overlay for readability"""
    selected_img_path = None

    # Try to find first existing image
    for fname in POSSIBLE_BG_IMAGES:
        path = os.path.join(ASSETS_DIR, fname)
        if os.path.isfile(path):
            selected_img_path = path
            break

    if selected_img_path:
        try:
            with open(selected_img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            css = f"""
            <style>
                [data-testid="stAppViewContainer"] {{
                    background-image: url("data:image/jpeg;base64,{encoded}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}

                /* Dark overlay ─ improves contrast a lot */
                [data-testid="stAppViewContainer"]::before {{
                    content: "";
                    position: absolute;
                    inset: 0;
                    background: rgba(0, 0, 0, 0.48);
                    z-index: -1;
                }}

                /* Force light text + shadow for better visibility */
                .stApp h1, .stApp h2, .stApp h3,
                .stApp p, .stApp div, .stApp span,
                .stApp .stMarkdown, label, .st-emotion-cache {{
                    color: #f8fafc !important;
                    text-shadow: 0 1px 3px rgba(0,0,0,0.9);
                }}

                /* Slightly transparent header */
                header {{
                    background: rgba(15, 23, 42, 0.7) !important;
                    backdrop-filter: blur(4px);
                }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not load background image → {e}")

    else:
        # Nice dark gradient fallback
        st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
            }
        </style>
        """, unsafe_allow_html=True)


@st.cache_data
def load_and_train():
    training_file = os.path.join(DATA_DIR, "Training.csv")
    training = pd.read_csv(training_file)
    cols = training.columns[:-1]
    X = training[cols]
    y = training["prognosis"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    reduced_data = training.groupby("prognosis").max()
    return clf, le, cols, reduced_data


@st.cache_data
def load_dictionaries():
    severity_dict = {}
    description_dict = {}
    precaution_dict = {}

    # Severity
    with open(os.path.join(DATA_DIR, "Symptom-severity.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                severity_dict[row[0]] = int(row[1])

    # Description
    with open(os.path.join(DATA_DIR, "symptom_Description.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header if exists
        for row in reader:
            if len(row) >= 2:
                description_dict[row[0]] = row[1]

    # Precautions
    with open(os.path.join(DATA_DIR, "symptom_precaution.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 5:
                precaution_dict[row[0]] = [p for p in row[1:] if p.strip()]

    return severity_dict, description_dict, precaution_dict


def check_pattern(symptom_list, user_input):
    cleaned = user_input.replace(" ", "_").strip()
    pattern = re.compile(re.escape(cleaned), re.IGNORECASE)
    matches = [s for s in symptom_list if pattern.search(s)]
    return (1, matches) if matches else (0, [])


def sec_predict(user_symptoms, all_columns):
    df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    X = df.iloc[:, :-1]
    y = df["prognosis"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=20)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    symptoms_dict = {sym: idx for idx, sym in enumerate(X.columns)}
    vec = np.zeros(len(symptoms_dict))

    for sym in user_symptoms:
        if sym in symptoms_dict:
            vec[symptoms_dict[sym]] = 1

    return model.predict([vec])[0]


def calc_condition(symptoms, days, severity_dict):
    if not symptoms:
        return "Please select some symptoms."
    total_severity = sum(severity_dict.get(s, 0) for s in symptoms)
    score = (total_severity * days) / (len(symptoms) + 1)
    if score > 13:
        return "You should see a doctor as soon as possible."
    return "Take precautions — condition does not appear severe."


def analyze_symptoms(clf, le, cols, reduced_data, initial_symptom, days, sev, desc, prec):
    tree_ = clf.tree_
    feature_name = [cols[i] if i >= 0 else "undefined" for i in tree_.feature]

    def walk_tree(node):
        if tree_.feature[node] >= 0:
            name = feature_name[node]
            val = 1 if name == initial_symptom else 0
            if val <= tree_.threshold[node]:
                return walk_tree(tree_.children_left[node])
            else:
                return walk_tree(tree_.children_right[node])
        else:
            node_values = tree_.value[node][0]
            idx = np.argmax(node_values)
            disease = le.inverse_transform([idx])[0]
            row = reduced_data.loc[disease]
            symptoms_present = list(row[row == 1].index)
            return symptoms_present, [disease]

    possible_symptoms, predicted_diseases = walk_tree(0)

    st.subheader("Select the symptoms you are experiencing:")
    selected_symptoms = []
    for sym in possible_symptoms:
        if st.checkbox(sym.replace("_", " ").title()):
            selected_symptoms.append(sym)

    if st.button("Get Diagnosis", type="primary"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        most_likely = sec_predict(selected_symptoms, cols)
        advice = calc_condition(selected_symptoms, days, sev)

        st.info(advice)
        st.success(f"**Possible condition:** {most_likely}")

        st.markdown("**Description**")
        st.write(desc.get(most_likely, "No description available."))

        st.markdown("**Recommended Precautions**")
        for p in prec.get(most_likely, ["No specific precautions found."]):
            if p.strip():
                st.write(f"• {p}")


def main():
    set_custom_style()
    initialize_user_file()

    if "view" not in st.session_state:
        st.session_state["view"] = "home"

    if st.session_state["view"] == "home":
        st.title("Disease Prediction System")
        st.markdown("Please login or register to continue.")
        col1, col2 = st.columns(2)
        if col1.button("Register", use_container_width=True):
            st.session_state["view"] = "register"
        if col2.button("Login", use_container_width=True):
            st.session_state["view"] = "login"

    elif st.session_state["view"] == "register":
        st.header("Create Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        phone = st.text_input("Phone Number")
        city = st.text_input("City")

        if st.button("Register"):
            if not username.strip():
                st.error("Username is required.")
            elif username_exists(username):
                st.error("Username already taken.")
            else:
                register_user(username, password, phone, city)

    elif st.session_state["view"] == "login":
        st.header("Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state["username"] = username
                st.session_state["view"] = "main"
            else:
                st.error("Invalid username or password.")

    elif st.session_state["view"] == "main":
        st.header(f"Welcome, {st.session_state['username']}")

        symptom_input = st.text_input("Type the main symptom you're experiencing:", key="symptom_input")

        if symptom_input:
            clf, le, cols, reduced_data = load_and_train()
            sev, desc, prec = load_dictionaries()

            found, matches = check_pattern(cols, symptom_input)

            if found:
                main_symptom = matches[0]   # take best match
                days = st.number_input("How many days have you had this symptom?", min_value=1, step=1)
                analyze_symptoms(clf, le, cols, reduced_data, main_symptom, days, sev, desc, prec)
            else:
                st.error("Sorry, we couldn't recognize that symptom. Try another spelling or a related term.")

        if st.button("Logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state["view"] = "home"
            st.rerun()


if __name__ == "__main__":
    main()
