import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

st.set_page_config(layout='wide', page_title=" Spam Message Classifier", page_icon="🚫")


st.markdown("""
    <style>
            
    .hero-section {
        background-image: url("https://www.shutterstock.com/image-vector/cybersecurity-warning-spam-sms-mobile-260nw-2714250287.jpg");
        background-size: cover;
        background-position: center;
        padding: clamp(30px, 8vw, 80px);
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        width: 100%;
    }

    .main-heading {
        font-size: clamp(2rem, 10vw, 4.5rem);
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
        text-shadow: 2px 2px 15px rgba(0,0,0,0.5);
    }
    
    .sub-text {
        font-size: clamp(1rem, 3vw, 1.5rem);
        opacity: 0.9;
        margin-top: 10px;
    }
    @media (max-width: 768px) {
        .hero-section {
            padding: 1.5rem;
            border-radius: 15px;
        }
        .main-heading {
            font-size: 2.2rem;
        }
        .sub-text {
            font-size: 1.1rem;
        }
        .stButton>button {
            width: 100%;
            font-size: 16px;
            height: 2.5em;
            border-radius: 10px;
            background-color: #ff4b4b;
            color: white;
        }
    }

    </style>
    
    <div class="hero-section">
        <h1 class="main-heading">🚫 Spam Message Classifier </h1>
        <p class="sub-text">AI-Powered Spam Massage Classifier System</p>
    </div>
    """, unsafe_allow_html=True)
    

def mycleaning(doc):
    return re.sub("[^a-zA-Z ]","",doc).lower()
    
model=joblib.load("Spam_model.pkl")


with st.sidebar:
    st.link_button("Food Sentiment Analysis","https://sentiment05.streamlit.app/")

    st.divider()

    st.image("spam.jpg", use_container_width=True)

    st.divider()
    
    st.title("🚀 About Project")
    st.info("This System uses Natural Language Processing (NLP) to automatically detect Spam Message.")
    
    st.title("🛠️ Technical Stack")
    st.write("**Streamlit:** Framework for creating the interactive web interface.")
    st.write("**Scikit-Learn:** Used for training and implementing the Sentiment Model.")
    st.write("**Pandas:** For data manipulation and reading CSV files.")
    st.write("**Numpy:** For mathematical operations and array processing.")
    st.write("**Joblib:** To load the pre-trained Machine Learning model.")
    st.write("**Regex (re):** For cleaning and preprocessing the text data.")
    
    st.divider()
    
    st.title("📞 Contact Us")
    st.success("📍 **AI Engineers @ DUCAT**")
    st.write("📧 **Email:** uditpal518@gmail.com")
    st.write("📱 **Phone:** +91 99999-88888")
    st.write("🌐 **Website:** www.ducatindia.com")


st.write("\n")
st.write("#### 🚫 Spam Message Detector")
sample=st.text_area("Enter Message...", placeholder="e.g. Win a Brand new iPhone 15! Click here to claim...", height=100)
if st.button("Predict",use_container_width=True):
    pred=model.predict([sample])
    prob=model.predict_proba([sample])
    if pred[0]=='ham':
        st.success("Valid 👍")
        st.write(f"Confidence Score : {prob[0][0]:.2f}")
        st.balloons()
    else:
        st.error("SPAM 👎") 
        st.write(f"Confidence Score : {prob[0][1]:.2f}")
        st.balloons()

st.divider()

st.write("#### Bulk Prediction")
data_source = st.radio("Choose Data Source:",["Upload your own file","Use Sample File"])
df = None
if data_source == "Upload your own file":
    file = st.file_uploader("Upload CSV & TEXT", type=["csv","txt"])
    if file:
        df = pd.read_csv(file, names=["Messages"])
elif data_source == "Use Sample File":
    with open("spam.txt", "br") as file:
        df = pd.read_csv(file, names=["Messages"])
if df is not None:
    placeholder=st.empty()
    placeholder.dataframe(df, use_container_width=True)
    if st.button("Predict",key="b2",use_container_width=True):
        corpus=df.MSG
        pred=model.predict(corpus)
        prob=np.max(model.predict_proba(corpus),axis=1)
        df['Msg Type']=pred
        df['Confidance']=prob
        placeholder.dataframe(df,use_container_width=True)
