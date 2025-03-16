import streamlit as st
import hashlib
import json
from pathlib import Path
import re
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards
from collections import Counter
import os

# Define the Decision Tree Node class
class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Predicted class (for leaf nodes)

# Define the Decision Tree class
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        elif isinstance(self.max_features, str) and self.max_features == 'sqrt':
            self.max_features = max(1, int(np.sqrt(self.n_features)))
        else:
            self.max_features = min(self.max_features, self.n_features)
        
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            # Leaf node
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
        
        # Randomly select a subset of features
        feature_idxs = np.random.choice(n_features, self.max_features, replace=False)
        
        # Find the best split
        best_feature_idx, best_threshold = self._best_split(X, y, feature_idxs)
        
        # Create child splits
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_threshold)
        
        # If the split doesn't yield any information gain, create a leaf node
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
        
        # Recursive calls to build the tree
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return DecisionTreeNode(best_feature_idx, best_threshold, left, right)
    
    def _best_split(self, X, y, feature_idxs):
        best_gain = -np.inf
        split_idx, split_threshold = None, None
        
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        # Calculate parent entropy
        parent_entropy = self._entropy(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self, y):
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        ps = ps[ps > 0]  # Remove zero probabilities
        return -np.sum(ps * np.log2(ps))
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Define the Random Forest class
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            
            # Bootstrap sampling (random sampling with replacement)
            n_samples = X.shape[0]
            bootstrap_idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idxs]
            y_bootstrap = y[bootstrap_idxs]
            
            # Train the tree on the bootstrap sample
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to get predictions per sample
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Majority voting
        return np.array([self._most_common_label(pred) for pred in tree_preds])
    
    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        probas = []
        for pred in tree_preds:
            # Count occurrences of each class
            class_counts = np.bincount(pred.astype(int), minlength=2)
            # Convert to probabilities
            proba = class_counts / len(pred)
            probas.append(proba)
            
        return np.array(probas)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

# File to store user credentials
USER_CREDENTIALS_FILE = "users.json"
CONTACT_RESPONSE_FILE = "D:/ESDPS_W_ALGO/response.xlsx"

# Load or create user credentials file
def load_user_data():
    if Path(USER_CREDENTIALS_FILE).exists():
        with open(USER_CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_user_data(users):
    with open(USER_CREDENTIALS_FILE, "w") as file:
        json.dump(users, file)

# Hashing passwords for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Password validation
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not any(char.isupper() for char in password):
        return "Password must contain at least one uppercase letter."
    if not any(char.islower() for char in password):
        return "Password must contain at least one lowercase letter."
    if not any(char.isdigit() for char in password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."
    return None

# Save contact form response to Excel
def save_contact_response(name, email, message):
    # Prepare data
    timestamp = pd.Timestamp.now()
    data = {"Timestamp": [timestamp], "Name": [name], "Email": [email], "Message": [message]}
    df = pd.DataFrame(data)
    
    # Check if file exists and append or create new
    if os.path.exists(CONTACT_RESPONSE_FILE):
        existing_df = pd.read_excel(CONTACT_RESPONSE_FILE)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(CONTACT_RESPONSE_FILE, index=False)
    else:
        df.to_excel(CONTACT_RESPONSE_FILE, index=False)

# Function to load the custom Random Forest model
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Function to make prediction and get probability using our custom model
def predict_diabetes(model, input_data):
    # Convert DataFrame to numpy array
    input_data_np = input_data.values
    
    # Get prediction
    prediction = model.predict(input_data_np)
    
    # Get probability of being diabetic (class 1)
    probas = model.predict_proba(input_data_np)
    prediction_prob = probas[0][1]  # Probability of class 1 (diabetic)
    
    return prediction, prediction_prob

# Recommendations based on the prediction probability
def get_recommendations(probability):
    if probability < 0.2:
        return """
        *Recommendations:*
        - 🌟 You are at low risk of diabetes. Continue maintaining a healthy lifestyle.
        - 🏃‍♂️ Regular exercise and a balanced diet are key to staying healthy.
        - 🩺 Monitor your health with regular check-ups.
        - 💧 Stay hydrated by drinking plenty of water.
        - 🚼 Ensure you get adequate sleep and manage stress effectively.
        """
    elif 0.2 <= probability < 0.4:
        return """
        *Recommendations:*
        - 😊 You are relatively safe, but it's good to stay vigilant.
        - 🥗 Maintain a healthy diet rich in vegetables, fruits, and whole grains.
        - 🚶‍♀️ Exercise regularly, aiming for at least 30 minutes most days of the week.
        - ⛔ Limit intake of processed foods and sugary drinks.
        - 🔬 Monitor your blood pressure and cholesterol levels regularly.
        - ⏳ Consider occasional fasting or intermittent fasting for metabolic health.
        """
    elif 0.4 <= probability < 0.6:
        return """
        *Recommendations:*
        - ⚠️ You are at moderate risk of diabetes.
        - 🍏 Pay close attention to your diet and exercise habits.
        - 🧺 Monitor your blood sugar levels regularly.
        - 🍞 Avoid excessive sugar and carbohydrate intake.
        - 🏋️‍♂️ Maintain a healthy weight and aim to reduce body fat percentage if necessary.
        - 🚶‍♀️ Stay active throughout the day, not just during exercise sessions.
        - 🧘‍♂️ Incorporate stress-reducing activities like yoga, meditation, or hobbies.
        """
    elif 0.6 <= probability < 0.8:
        return """
        *Recommendations:*
        - 🚨 You are at high risk of diabetes.
        - 🩺 Consult with a healthcare provider for personalized advice.
        - 🥦 Make significant lifestyle changes to reduce risk, such as increasing physical activity and improving your diet.
        - 🧺 Monitor your blood sugar levels frequently.
        - 🥗 Reduce intake of high glycemic index foods and increase fiber intake.
        - 💝 Stay away from tobacco and limit alcohol consumption.
        - 🏃‍♂️ Engage in regular physical activities that you enjoy to ensure consistency.
        - 👩‍⚕️ Consider working with a nutritionist or dietitian for tailored dietary plans.
        """
    elif probability >= 0.8:
        return """
        *Recommendations:*
        - 🔴 You are at very high risk of diabetes.
        - 🩺 Seek immediate medical advice and intervention.
        - 🏋️‍♂️ Follow a strict diet and exercise regimen as advised by your healthcare provider.
        - 🧺 Monitor your blood sugar levels very closely.
        - 🧑‍🧑‍🧑 Consider joining a support group for individuals at risk of or managing diabetes.
        - 🧘‍♀️ Take proactive steps to manage stress and ensure mental well-being.
        - 📅 Regularly review and adjust your lifestyle habits in consultation with your healthcare team.
        """

# Custom CSS for UI enhancement
def apply_custom_css():
    st.markdown(
        """
        <style>
        .main-header {
            color: #4CAF50;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            height: 3em;
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        .st-badge-number {
            background-color: #f39c12;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function for authentication and app
def main():
    apply_custom_css()
    
    # Session state for managing authentication
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    if not st.session_state['logged_in']:
        # Show the login/signup UI first
        st.sidebar.title("Authentication")
        menu = st.sidebar.radio("Menu", ["Login", "Sign Up"])

        users = load_user_data()

        if menu == "Sign Up":
            st.subheader("Sign Up")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Sign Up"):
                if username in users:
                    st.error("Username already exists. Please choose a different one.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    validation_error = validate_password(password)
                    if validation_error:
                        st.error(validation_error)
                    else:
                        users[username] = hash_password(password)
                        save_user_data(users)
                        st.success("You have successfully signed up! Please proceed to Login.")

        elif menu == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                hashed_password = hash_password(password)
                if username in users and users[username] == hashed_password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username  # Store username in session state
                    st.success(f"Welcome, {username}!")
                    st.rerun()  # Re-run to load the main app after login
                else:
                    st.error("Invalid username or password.")
    else:
        # Display signed-in user info and sign-out button in sidebar
        st.sidebar.title("User Info")
        st.sidebar.write(f"You are signed in as: **{st.session_state['username']}**")
        if st.sidebar.button("Sign Out"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.success("You have been signed out.")
            st.rerun()

        # Main navigation after authentication
        menu = st.sidebar.radio("Navigation", ["Welcome", "Predict Diabetes", "Dashboard", "About", "Contact"])

        if menu == "Welcome":
            st.markdown('<div class="main-header">Welcome to DiaSense : An Early Stage Diabetes Prediction System</div>', unsafe_allow_html=True)
            st.image("D:/ESDPS_W_ALGO/assets/banner.jpg", use_container_width=True)
            st.write("""
            DiaSense uses machine learning to predict diabetes risk based on early symptoms. 
            It helps you take charge of your health with preventive insights.
            DiaSense leverages advanced machine learning technology to provide you with a precise assessment of your risk for diabetes. Our system analyzes your health data to offer personalized predictions
            based on early diabetes symptoms helping you make informed decisions about your well-being.
            """)
        elif menu == "Predict Diabetes":
            st.markdown('<div class="main-header">DiaSense : An Early Stage Diabetes Prediction System</div>', unsafe_allow_html=True)
            
            # Create input fields for user data
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            polyuria = st.selectbox("Polyuria (Excessive urination)", ["Yes", "No"])
            polydipsia = st.selectbox("Polydipsia (Excessive thirst)", ["Yes", "No"])
            sudden_weight_loss = st.selectbox("Sudden Weight Loss (Unexplained weight loss)", ["Yes", "No"])
            weakness = st.selectbox("Weakness (Fatigue or lack of energy)", ["Yes", "No"])
            polyphagia = st.selectbox("Polyphagia (Excessive hunger)", ["Yes", "No"])
            genital_thrush = st.selectbox("Genital Thrush (Fungal infection around genital area)", ["Yes", "No"])
            visual_blurring = st.selectbox("Visual Blurring (Difficulty seeing clearly)", ["Yes", "No"])
            itching = st.selectbox("Itching (Persistent itchiness)", ["Yes", "No"])
            irritability = st.selectbox("Irritability (Mood swings or frustration)", ["Yes", "No"])
            delayed_healing = st.selectbox("Delayed Healing (Wounds that take time to heal)", ["Yes", "No"])
            partial_paresis = st.selectbox("Partial Paresis (Weakness in specific muscles)", ["Yes", "No"])
            muscle_stiffness = st.selectbox("Muscle Stiffness (Tight or rigid muscles)", ["Yes", "No"])
            alopecia = st.selectbox("Alopecia (Hair loss)", ["Yes", "No"])
            obesity = st.selectbox("Obesity (High body weight/BMI)", ["Yes", "No"])

            # Convert input data to pandas DataFrame with feature names
            input_columns = [
                "age", "gender", "polyuria", "polydipsia", "sudden weight loss", "weakness",
                "polyphagia", "genital thrush", "visual blurring", "itching", "irritability", "delayed healing",
                "partial paresis", "muscle stiffness", "alopecia", "obesity"
            ]
            input_data = pd.DataFrame([[
                age,
                1 if gender == "Male" else 0,
                1 if polyuria == "Yes" else 0,
                1 if polydipsia == "Yes" else 0,
                1 if sudden_weight_loss == "Yes" else 0,
                1 if weakness == "Yes" else 0,
                1 if polyphagia == "Yes" else 0,
                1 if genital_thrush == "Yes" else 0,
                1 if visual_blurring == "Yes" else 0,
                1 if itching == "Yes" else 0,
                1 if irritability == "Yes" else 0,
                1 if delayed_healing == "Yes" else 0,
                1 if partial_paresis == "Yes" else 0,
                1 if muscle_stiffness == "Yes" else 0,
                1 if alopecia == "Yes" else 0,
                1 if obesity == "Yes" else 0
            ]], columns=input_columns)

            # Load model and predict
            model_path = 'D:/ESDPS_W_ALGO/ML_MODEL/random_forest_model.pkl'
            
            if st.button("Predict Diabetes"):
                with st.spinner('Loading model and predicting...'):
                    try:
                        # Load the custom Random Forest model
                        model = load_model(model_path)
                        time.sleep(2)  # Simulate loading
                        
                        # Make prediction
                        prediction, prediction_prob = predict_diabetes(model, input_data)
                        
                        # Display results
                        st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
                        st.metric("Prediction Probability", f"{prediction_prob * 100:.2f}%")
                        recommendations = get_recommendations(prediction_prob)
                        st.markdown(recommendations)
                        
                        # Plotting the result for better visualization
                        fig = px.pie(names=['Diabetic', 'Non-Diabetic'], 
                                    values=[prediction_prob, 1 - prediction_prob], 
                                    title='Diabetes Prediction Probability',
                                    color_discrete_sequence=['#ff6361', '#58508d'])
                        st.plotly_chart(fig)
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error(f"Please ensure the model file exists at {model_path}")

        elif menu == "Dashboard":
            st.markdown('<div class="main-header">User Health Dashboard</div>', unsafe_allow_html=True)
            st.write("""
            This dashboard presents your past predictions and health trends to provide insight into the progress made.
            """)
            # Simulated Data for Past Predictions
            data = {'Date': pd.date_range(start='2024-01-01', periods=10, freq='M'),
                    'Diabetes Risk (%)': [12, 25, 33, 29, 45, 55, 62, 48, 37, 50]}
            df = pd.DataFrame(data)
            st.line_chart(df.set_index('Date'))
            st.write("Trends in Diabetes Prediction Risk")

        elif menu == "About":
            st.title("About DiaSense")
            st.write("""
                At DiaSense, we are at the forefront of diabetes care, combining advanced technology with a passion for health and well-being. 
                Our mission is to provide individuals with precise and actionable insights into their diabetes risk, helping them make informed decisions about their health.
                
                **DiaSense provides diabetes risk predictions based on symptoms and health data.** Developed with a focus on preventive care, it empowers users to take proactive steps in managing their health.
            """)
            
            st.subheader("Types of Diabetes")
            st.write("### Type 1 Diabetes")
            st.write("An autoimmune condition where the body attacks insulin-producing cells in the pancreas, leading to high blood sugar levels.")
            st.write("### Type 2 Diabetes")
            st.write("A metabolic disorder characterized by insulin resistance and eventual insulin deficiency, often linked to lifestyle factors and obesity.")
            st.write("### Gestational Diabetes")
            st.write("A type of diabetes that occurs during pregnancy and typically resolves after childbirth, but may increase the risk of developing Type 2 diabetes later.")
            
            st.subheader("Symptoms of Diabetes")
            symptoms = {
                "Polyuria": "Excessive urination, often one of the first signs of diabetes.",
                "Polydipsia": "Excessive thirst, another common symptom of diabetes.",
                "Sudden Weight Loss": "Unexplained weight loss can be a symptom of diabetes.",
                "Weakness": "A lack of energy and general fatigue.",
                "Polyphagia": "Excessive hunger, even after eating.",
                "Genital Thrush": "Fungal infection in the genital area.",
                "Visual Blurring": "Difficulty in seeing clearly.",
                "Itching": "Persistent itchiness across the body.",
                "Irritability": "Mood swings and increased frustration.",
                "Delayed Healing": "Cuts or wounds that take longer to heal.",
                "Partial Paresis": "Weakness in specific muscles.",
                "Muscle Stiffness": "Tight or rigid muscles causing discomfort.",
                "Alopecia": "Hair loss, especially noticeable on the scalp.",
                "Obesity": "Having a high body weight or Body Mass Index (BMI)."
            }
            
            for symptom, description in symptoms.items():
                st.write(f"**{symptom}**: _{description}_")
            
            st.subheader("FAQs")
            st.write("Click on a question to reveal the answer.")

            with st.expander("What is DiaSense?"):
                st.write("""
                    DiaSense is a diabetes risk prediction tool that uses advanced machine learning models and user-provided data to assess diabetes risk levels. It aims to promote preventive care and empower individuals with actionable health insights.
                """)

            with st.expander("How does DiaSense predict diabetes risk?"):
                st.write("""
                    DiaSense analyses user-inputted health data, such as symptoms and lifestyle factors, using a machine learning model trained on relevant datasets. This process enables an accurate risk prediction.
                """)

            with st.expander("Is DiaSense a replacement for medical diagnosis?"):
                st.write("""
                    No, DiaSense is not a replacement for professional medical diagnosis. It provides risk predictions to encourage early detection and preventive care. Always consult with a healthcare professional for an official diagnosis.
                """)

            with st.expander("What data is required for DiaSense to provide predictions?"):
                st.write("""
                    DiaSense requires basic health data, including symptoms, age, weight, family history, and lifestyle factors. The accuracy of predictions depends on the quality and completeness of the input data.
                """)
            
            with st.expander("Can DiaSense help in managing diabetes?"):
                st.write("""
                    While DiaSense focuses on risk prediction and prevention, it can provide valuable insights that help individuals monitor potential risks. For ongoing diabetes management, consult with a healthcare provider.
                """)

        elif menu == "Contact":
            st.title("Contact Us")

            # Contact details
            st.write("For any inquiries, support, or feedback, feel free to reach us through the following:")
            
            # Email and Phone
            st.write("**Email**: diasense2024@gmail.com")
            st.write("**Phone**: +977 98080000000")
            
            # Optional: Contact Form for user input
            st.subheader("Send Us a Message")
            
            # User inputs for message
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            
            # Button to submit the form
            if st.button("Submit"):
                if name and email and message:
                    save_contact_response(name, email, message)
                    st.success("Thank you for reaching out! Your message has been recorded.")
                else:
                    st.error("Please fill out all fields before submitting.")

if __name__ == "__main__":
    main()