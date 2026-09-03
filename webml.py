# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Set up the Streamlit app title and description
st.write("""
# Parkinson's Disease Detection
Detect if someone has parkinson's using machine learning !
""")
# Display an image
image = Image.open('headerB.jpg')
st.image(image, caption="ML", width="stretch")
# Read the dataset
df = pd.read_csv("parkinsons.csv")

df.head()
df.info()

# Display the dataset information and summary statistics
st.subheader('Data Information:')
# show data as a table
st.dataframe(df)
# show statistics on the data
st.write(df.describe())
# show the data as a line chart
chart = st.line_chart(df.drop(columns=['name'], axis=1))

# Check for missing values in the dataset
df.isnull().sum()
# Display the count of each target class
df['status'].value_counts()
# Display the mean values grouped by target class
df.groupby('status').mean()
# Split the dataset into features (X) and target (Y)
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

# Spilt the data set in 80% Training and 20% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Display the shapes of the data arrays
st.write(X.shape, X_train.shape, X_test.shape)

# Creating a sidebar for user input
def get_user_input():
    MDVP_Fo = st.sidebar.slider('MDVP:Fo(Hz)', 88.333, 260.105, 107.332)
    MDVP_Fhi = st.sidebar.slider('MDVP:Fhi(Hz)', 102.145, 592.03, 157.302)
    MDVP_Flo = st.sidebar.slider('MDVP:Flo(Hz)', 65.476, 239.17, 74.997)
    MDVP_Jitter = st.sidebar.slider('MDVP:Jitter(%)', 0.00168, 0.03316, 0.00505)
    MDVP_Jitter_Abs = st.sidebar.slider('MDVP:Jitter(Abs)', 0.000007, 0.00026, 0.00011)
    MDVP_RAP = st.sidebar.slider('MDVP:RAP', 0.00068, 0.02144, 0.00696)
    MDVP_PPQ = st.sidebar.slider('MDVP:PPQ ', 0.00092, 0.01958, 0.00183)
    Jitter_DDP = st.sidebar.slider('Jitter:DDP', 0.00204, 0.06433, 0.01394)
    MDVP_Shimmer = st.sidebar.slider('MDVP:Shimmer', 0.00954, 0.11908, 0.01608)
    MDVP_Shimmer_DB = st.sidebar.slider('MDVP:Shimmer(dB)', 0.085, 1.302, 0.255)
    Shimmer_APQ3 = st.sidebar.slider('Shimmer:APQ3', 0.00455, 0.05647, 0.01073)
    Shimmer_APQ5 = st.sidebar.slider('Shimmer:APQ5', 0.0057, 0.0794, 0.0313)
    MDVP_APQ = st.sidebar.slider('MDVP:APQ ', 0.00719, 0.13778, 0.04368)
    Shimmer_DDA = st.sidebar.slider('Shimmer:DDA', 0.01364, 0.16942, 0.03218)
    NHR = st.sidebar.slider('NHR', 0.00065, 0.31482, 0.01353)
    HNR = st.sidebar.slider('HNR', 8.441, 33.047, 17.536)
    RPDE = st.sidebar.slider('RPDE', 0.25657, 0.685151, 0.360148)
    DFA = st.sidebar.slider('DFA', 0.574282, 0.825288, 0.223797)
    spread1 = st.sidebar.slider('spread1', -7.96498, -2.43403, -6.966321)
    spread2 = st.sidebar.slider('spread2', 0.006274, 0.450493, 0.311173)
    D2 = st.sidebar.slider('D2', 1.423287, 3.671155, 1.91399)
    PPE = st.sidebar.slider('PPE', 0.044539, 0.527367, 0.015961)
    # Get the user's input
    user_data = {
        'MDVP_Fo(Hz)': MDVP_Fo,
        'MDVP_Fhi(Hz)': MDVP_Fhi,
        'MDVP_Flo(Hz)': MDVP_Flo,
        'MDVP_Jitter(%)': MDVP_Jitter,
        'MDVP_Jitter(Abs)': MDVP_Jitter_Abs,
        'MDVP_RAP': MDVP_RAP,
        'MDVP_PPQ': MDVP_PPQ,
        'Jitter_DDP': Jitter_DDP,
        'MDVP_Shimmer': MDVP_Shimmer,
        'MDVP_Shimmer(dB)': MDVP_Shimmer_DB,
        'Shimmer_APQ3': Shimmer_APQ3,
        'Shimmer_APQ5': Shimmer_APQ5,
        'MDVP_APQ': MDVP_APQ,
        'Shimmer_DDA': Shimmer_DDA,
        'NHR': NHR,
        'HNR': HNR,
        'RPDE': RPDE,
        'DFA': DFA,
        'spread1': spread1,
        'spread2': spread2,
        'D2': D2,
        'PPE': PPE
    }

    # Transform data into data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the use input into variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader('User Input: ')
st.write(user_input)
# Creating columns to display accuracy score for both Algorithms
col1, col2 = st.columns(2)

# Create and Train the Random Forest Classifier
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

with col1:
    # Show the model's metrics and Display the accuracy score of the Random Forest Classifier
    st.subheader('RandomForestClassifier Accuracy Score')
    st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the models prediction in a variable
    prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
    st.subheader('Classification: ')
    st.write(prediction)
    if prediction == 0:
        st.write('<p style="font-size:30px;"> 😊 Negative, No Parkinsons found<p/>', unsafe_allow_html=True)
    else:
        st.write('<p style="font-size:30px;"> 😢 Positive, Parkinsons found<p/>', unsafe_allow_html=True)

# Train a Logistic Regression model
LogisticRegression = LogisticRegression(C=0.4, max_iter=1000, solver='liblinear')
LogisticRegression.fit(X_train, Y_train)

with col2:
    st.subheader('Logistic Regression Accuracy Score')
    st.write(str(accuracy_score(Y_test, LogisticRegression.predict(X_test)) * 100) + '%')
    # Display the accuracy score of the Logistic Regression model
    pred = LogisticRegression.predict(user_input)

    # Set a subheader and display the classification
    st.subheader('Classification2: ')
    st.write(pred)
    if pred == 0:
        st.write('<p style="font-size:30px;"> 😊 Negative, No Parkinsons found<p/>', unsafe_allow_html=True)
    else:
        st.write('<p style="font-size:30px;"> 😢 Positive, Parkinsons found<p/>', unsafe_allow_html=True)
