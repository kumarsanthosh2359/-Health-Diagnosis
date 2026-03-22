import streamlit as st
import os
import csv
import base64
import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

IMG_PATH = os.path.join(BASE_DIR, "sg.jpg")
USER_DB_FILE = os.path.join(DATA_DIR, "users_db.csv")

# ===============================
# INIT FILES
# ===============================
def initialize_user_file():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password", "phone", "city"])


# ===============================
# BACKGROUND
# ===============================
def set_background():
    if os.path.exists(IMG_PATH):
        with open(IMG_PATH, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)


# ===============================
# USER SYSTEM
# ===============================
def username_exists(username):
    if not os.path.exists(USER_DB_FILE):
        return False

    with open(USER_DB_FILE, "r") as f:
        reader = csv.DictReader(f)
        return any(row["username"] == username for row in reader)


def register_user(username, password, phone, city):
    with open(USER_DB_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password, phone, city])

    st.success("Registered Successfully!")
    st.session_state["view"] = "main"
    st.session_state["username"] = username


def login_user(username, password):
    if not os.path.exists(USER_DB_FILE):
        return False

    with open(USER_DB_FILE, "r") as f:
        reader = csv.DictReader(f)
        return any(row["username"] == username and row["password"] == password for row in reader)


# ===============================
# LOAD MODEL
# ===============================
@st.cache_data
def load_and_train():
    path = os.path.join(DATA_DIR, "Training.csv")

    if not os.path.exists(path):
        st.error("Training.csv missing in data folder")
        return None, None, None, None

    df = pd.read_csv(path)

    X = df.iloc[:, :-1]
    y = df["prognosis"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model, le, X.columns, df.groupby("prognosis").max()


# ===============================
# LOAD DICTIONARIES
# ===============================
@st.cache_data
def load_dict():
    sev, desc, prec = {}, {}, {}

    try:
        with open(os.path.join(DATA_DIR, "Symptom-severity.csv")) as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                sev[r[0]] = int(r[1])

        with open(os.path.join(DATA_DIR, "symptom_Description.csv")) as f:
            reader = csv.reader(f)
            for r in reader:
                desc[r[0]] = r[1]

        with open(os.path.join(DATA_DIR, "symptom_precaution.csv")) as f:
            reader = csv.reader(f)
            for r in reader:
                prec[r[0]] = r[1:]

    except:
        st.warning("Dictionary files missing")

    return sev, desc, prec


# ===============================
# HELPERS
# ===============================
def check_pattern(cols, inp):
    inp = inp.replace(" ", "_")
    return [c for c in cols if inp in c]


def calc_condition(exp, days, sev):
    score = sum(sev.get(i, 0) for i in exp) * days
    return "You should see a doctor." if score > 10 else "Take precautions."


# ===============================
# MAIN APP
# ===============================
def main():
    set_background()
    initialize_user_file()

    if "view" not in st.session_state:
        st.session_state["view"] = "home"

    # HOME
    if st.session_state["view"] == "home":
        st.title("Health Diagnosis System")

        if st.button("Login"):
            st.session_state["view"] = "login"

        if st.button("Register"):
            st.session_state["view"] = "register"

    # LOGIN
    elif st.session_state["view"] == "login":
        st.header("Login")

        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.session_state["username"] = u
                st.session_state["view"] = "main"
            else:
                st.error("Invalid Login")

    # REGISTER
    elif st.session_state["view"] == "register":
        st.header("Register")

        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ph = st.text_input("Phone")
        c = st.text_input("City")

        if st.button("Submit"):
            if username_exists(u):
                st.error("Username exists")
            else:
                register_user(u, p, ph, c)

    # MAIN APP
    elif st.session_state["view"] == "main":
        st.header(f"Welcome {st.session_state['username']}")

        model, le, cols, data = load_and_train()
        sev, desc, prec = load_dict()

        symptom = st.text_input("Enter symptom")

        if symptom and model is not None:
            matches = check_pattern(cols, symptom)

            if matches:
                st.subheader("Select symptoms:")

                selected = []

                # CHECKBOX UI (3 columns)
                col1, col2, col3 = st.columns(3)
                col_list = [col1, col2, col3]

                for i, sym in enumerate(matches):
                    if col_list[i % 3].checkbox(sym):
                        selected.append(sym)

                days = st.number_input("Days", 1)

                if st.button("Diagnose"):
                    if not selected:
                        st.warning("Select at least one symptom")
                    else:
                        input_vector = np.isin(cols, selected).astype(int)
                        result = model.predict([input_vector])
                        disease = le.inverse_transform(result)[0]

                        st.success(f"Disease: {disease}")
                        st.info(calc_condition(selected, days, sev))
                        st.write(desc.get(disease, "No description"))
                        st.write(prec.get(disease, []))
            else:
                st.error("Symptom not found")

        if st.button("Logout"):
            st.session_state["view"] = "home"


if __name__ == "__main__":
    main()
