import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import random
import io
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords


import tweepy
# Download stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ✅ Set Page Config FIRST to avoid Streamlit API Exception
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

# ✅ Load Tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

# ✅ Download Model from Google Drive
@st.cache_resource
def download_model():
    # Google Drive link (replace with your actual link)
    drive_link = "https://drive.google.com/drive/folders/1zUEzBJeYlMatNQgpeeM7qQAyU5-ntEmw?usp=drive_link"
    model_file = "text_classification_model.h5"

    # Download the model if it doesn't already exist
    if not os.path.exists(model_file):
        gdown.download(drive_link, model_file, quiet=False)
    
    return model_file

# ✅ Load Pretrained Sentiment Model
@st.cache_resource
def load_sentiment_model():
    model_file = download_model()  # Ensure model is downloaded
    return load_model(model_file)

# ✅ Load Tokenizer & Model
tokenizer = load_tokenizer()
model = load_sentiment_model()

# ✅ Define Preprocessing Function
def preprocess_text(text):
    text=text.lower()             # convert to lowercase
    text=re.sub(r'@\w+',"",text)   # remove mentios
    text = re.sub(r'http\S+', '', text)     # remove url
    text=re.sub(r"[^a-zA-Z\s]","",text)  #remove punctuation
    words = text.split()   #tokenization
    words=[word for word in words if word not in stop_words]       #remove stopwords
    return " ".join(words)
    

# ✅ Function to Predict Sentiment
def predict_tweet_sentiment(tweet):
    processed_tweet = preprocess_text(tweet)
    sequence = tokenizer.texts_to_sequences([processed_tweet]) 
    max_len = 50
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')  # Adjust max_len
    prediction = model.predict(padded_sequence)
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    return sentiment, float(prediction[0][0])

# ✅ Main Function for Sidebar Navigation
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Tweet Analysis", "Upload Tweets Dataset",
                                           "Fetch Live Tweets", "About"])

    if selection == "Home":
        show_home()
    elif selection == "Tweet Analysis":
        show_live_tweet_analysis()
    elif selection == "Upload Tweets Dataset":
        show_upload_dataset()
    elif selection == "Fetch Live Tweets":
        fetch_live_tweets()
    elif selection == "About":
        show_about()

# ✅ Home Page
def show_home():
    st.title("Welcome to the Twitter Sentiment Analyzer")
    st.write("""
        This application fetches live tweets from Twitter/X and analyzes their sentiment.
        You can also upload your own dataset of reviews to get insights on sentiment distribution.
    """)
    
    # ✅ Display Image
    image = Image.open("performing-twitter-sentiment-analysis1.jpg")
    st.image(image, caption="App Logo", use_column_width=True)

    st.subheader("Use the sidebar to navigate different sections of the application.")

# ✅ Live Tweet Analysis Page
def show_live_tweet_analysis():
    st.title("📢 Live Tweet Sentiment Analysis")

    st.write("Analyze the sentiment of a single tweet in real time!")

    user_tweet = st.text_area("✍️ Enter a tweet:", placeholder="Type a tweet here...")

    if st.button("🔍 Analyze Sentiment"):
        if user_tweet.strip():
            sentiment, score = predict_tweet_sentiment(user_tweet)

            # 🔹 **1. Display Sentiment with an Emoji**
            st.markdown(f"## Sentiment: {sentiment}")
            
            # 🔹 **2. Probability Score as a Progress Bar**
            st.markdown("### 📊 Sentiment Confidence Score")
            # Define progress bar color dynamically
            progress_color = "green" if score > 0.6 else "red"
            # Show sentiment score with progress bar
            st.progress(int(score * 100)) 
            st.metric(label="Confidence Score", value=f"{score:.2%}", delta=random.uniform(-0.02, 0.02))
            # Custom styled message
            if score > 0.8:
                st.success("🔵 Highly Confident Prediction!")
            elif score > 0.5:
                st.info("🟡 Moderate Confidence")
            else:
                st.warning("🔴 Low Confidence")


            # 🔹 **4. Generate Word Cloud**
            st.markdown("### ☁️ Word Cloud of Tweet")
            preprocessed_tweet = preprocess_text(user_tweet)
            wordcloud = WordCloud(width=500, height=300, background_color="black").generate(preprocessed_tweet)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # 🔹 **5. Historical Sentiment Tracker (Dummy Data)**
            st.markdown("### 📈 Sentiment History")
            history_data = pd.DataFrame({
                "Tweet Number": list(range(1, 11)),
                "Sentiment Score": np.round(np.random.uniform(0, 1, 10), 2)
            })
            fig = px.line(history_data, x="Tweet Number", y="Sentiment Score",
                          title="Sentiment Score Over Time",
                          markers=True, line_shape="spline")
            st.plotly_chart(fig)

        else:
            st.warning("⚠️ Please enter a tweet to analyze.")


# ✅ Upload Tweets Dataset Page (Future Feature)
# Function to read CSV or Excel
def read_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1]

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format! Please upload a CSV or Excel file.")
        return None

    if "text" not in df.columns:
        st.error("The uploaded file must contain a 'text' column.")
        return None

    return df

# Function to upload and analyze dataset
def show_upload_dataset():
    st.title("📊 Bulk Tweet Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file with tweets", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = read_file(uploaded_file)

        if df is None:
            return  # Stop execution if file is invalid

        # Show first 5 rows before processing
        st.write("### First 5 Rows of Uploaded File:")
        st.dataframe(df.head())

        # Apply sentiment analysis
        df["Sentiment"], df["Score"] = zip(*df["text"].apply(predict_tweet_sentiment))

        # Show first 5 rows after processing
        st.write("### First 5 Rows After Sentiment Analysis:")
        st.dataframe(df.head())

                # Plot sentiment distribution
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Sentiment Distribution",
            color="Sentiment",
            text="Count",
            color_discrete_map={"Positive 😊": "green", "Negative 😞": "red"}
        )
        st.plotly_chart(fig)

        # Convert back to original format for download
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "csv":
            output_file = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions (CSV)", output_file, "predicted_tweets.csv", "text/csv")
        else:
            # Use BytesIO for Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            excel_buffer.seek(0)
            st.download_button("Download Predictions (Excel)", excel_buffer, "predicted_tweets.xlsx","application/vnd.openxmlformatsofficedocument.spreadsheetml.sheet")    



# Function to fetch tweets using Twitter API v2
def fetch_tweets(api_key, api_secret, access_token, access_token_secret, bearer_token, query, count=10):
    try:
        # Authenticate Twitter API v2
        client = tweepy.Client(bearer_token=bearer_token)

        # Fetch tweets
        tweets = client.search_recent_tweets(query=query, max_results=count, tweet_fields=["created_at"])

        tweet_list = [{"Tweet": tweet.text, "Time": tweet.created_at} for tweet in tweets.data]

        return pd.DataFrame(tweet_list)
    
    except Exception as e:
        st.error(f"❌ Error fetching tweets: {e}")
        return None

# Streamlit UI for fetching live tweets
def fetch_live_tweets():
    st.title("🔴 Fetch Live Tweets in Real-time")

    with st.sidebar:
        st.subheader("🔑 Enter Twitter API Keys")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        access_token = st.text_input("Access Token", type="password")
        access_token_secret = st.text_input("Access Token Secret", type="password")
        bearer_token = st.text_input("Bearer Token", type="password")

    st.subheader("📢 Search for Tweets")
    query = st.text_input("Enter a keyword or hashtag:")
    count = st.slider("Number of tweets to fetch:", 5, 50, 10)

    if st.button("🚀 Fetch Tweets"):
        if all([api_key, api_secret, access_token, access_token_secret, bearer_token, query]):
            tweets_df = fetch_tweets(api_key, api_secret, access_token, access_token_secret, bearer_token, query, count)

            if tweets_df is not None and not tweets_df.empty:
                st.success(f"✅ Successfully fetched {len(tweets_df)} tweets for '{query}'")
                st.dataframe(tweets_df)

                # Download option
                csv = tweets_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv, "live_tweets.csv", "text/csv")
            else:
                st.warning("⚠️ No tweets found. Try a different keyword.")
        else:
            st.warning("⚠️ Please enter all API keys and a search query.")


# ✅ About Page
def show_about():
    st.title("About This Application")
    st.write("""
        This Twitter Sentiment Analyzer uses an LSTM model trained on tweets to predict sentiments.
        You can analyze individual tweets, upload datasets, and visualize sentiment trends.
    """)

import streamlit as st

def show_about():
    st.title("About This App")
    
    st.markdown("""
    ## 📝 **Twitter Sentiment Analysis**
    This application analyzes the sentiment of tweets using a **deep learning model (LSTM)**. 
    It can process:
    
    - **Single tweets** (Live Tweet Analysis) 📢  
    - **Bulk tweets** from uploaded CSV/Excel files 📂  

    The app predicts whether a tweet expresses **Positive 😊 or Negative 😞** sentiment.

    ---
    
    ## 🛠 **Technologies Used**
    - **Natural Language Processing (NLP)** for text processing  
    - **TensorFlow/Keras (LSTM Model)** for sentiment classification  
    - **Streamlit** for building an interactive UI  
    - **Plotly** for data visualizations  
    - **Pandas & NumPy** for data manipulation  

    ---

    ## 📊 **Data & Model Information**
    - The model was trained on the **Hugging Face Tweets Dataset (180,000 tweets)**  
    - Achieved **83% accuracy** on test data  
    - Preprocessing steps include **tokenization, padding, and stopword removal**  

    ---

    ## 👨‍💻 **Developer**
    - **[Sachin Balhara]** – Data Scientist & AI Developer  
    - **GitHub:** [https://github.com/SACHINBALHARA]  
    - **LinkedIn:** [www.linkedin.com/in/sachin-balhara-05a084271]  
    - **Website:** [Your Website](#)  

    ---
    
    🚀 *This project is open-source! Feel free to contribute & improve it.* 🎯
    """)


# ✅ Run the Streamlit App
if __name__ == "__main__":
    main()
