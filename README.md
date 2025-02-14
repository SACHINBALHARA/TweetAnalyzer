# **Twitter Sentiment Analysis Web App**  

This project is a web application for performing sentiment analysis on tweets using a trained text classification model. The app is built with **Python** and leverages **Machine Learning, LSTM, Bidirectional Learning**, and **NLP** tools to classify sentiments as positive, negative, or neutral.  

---

## **📂 Project Structure**  

Here's an overview of the files in the repository:  

- `app1.py` - The main application script that runs the web app using a web framework (e.g., Flask or Streamlit).  
- `performing-twitter-sentiment-analysis1.jpg` - An image demonstrating the sentiment analysis process or used in the app UI.  
- `preprocess_text.pkl` - A serialized Python object used for text preprocessing (e.g., tokenization, stopword removal).  
- `preprocessing_info.json` - A JSON file containing configuration details or metadata for preprocessing.  
- `test_tweets.csv` - A CSV file containing sample tweets for testing the sentiment analysis model.  
- `tweets_test.xlsx` - An Excel file containing additional test data for sentiment analysis.  

**External File:**  
- `text_classification_model.h5` - A trained Keras model for text classification (stored on Google Drive due to size constraints).  

---

## **🔧 Skills and Technologies Used**  

The project incorporates the following skills and technologies:  

- **Python** - Core programming language for the project.  
- **Machine Learning** - Used to train and evaluate the sentiment analysis model.  
- **LSTM (Long Short-Term Memory)** - A type of recurrent neural network (RNN) used for text classification.  
- **Bidirectional Learning** - Enhances the model’s ability to understand text by reading it in both directions (forward and backward).  
- **NLP (Natural Language Processing)** - Used for text preprocessing and sentiment analysis.  
- **NLP Tools**:  
  - **Tokenization** - Splits text into words or tokens.  
  - **Stopword Removal** - Removes common words (e.g., "the", "is") to focus on meaningful words.  
  - **Text Cleaning** - Cleans the input text (removing special characters, converting to lowercase, etc.).  

---

## **🚀 Features**  

The web app comes with the following capabilities for users:  

- **Home** - Navigate to the home page of the application.  
- **Tweet Analysis** - Enter individual tweets and classify their sentiments as positive, negative, or neutral.  
- **Upload Tweets Dataset** - Upload a CSV or Excel file of tweets and perform batch sentiment analysis.  
- **Fetch Live Tweets** - Fetch real-time tweets from Twitter for sentiment analysis (requires Twitter API integration).  
- **About** - Provides details about the app, its purpose, and usage instructions.  

---

## **🛠 Installation**  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/your-repo-name.git  
   cd your-repo-name  

2. Install the required Python packages:
   '''bash
   pip install -r requirements.txt

3. Download the trained model (text_classification_model.h5) from the link below and place it in the root of the repository.
 ***Download Model from Google Drive***

## **🚀 How to Run the App**
1. Ensure the model file (text_classification_model.h5) is in the project directory.

2. Run the application using Python:
   '''bash
   python app1.py

3. Open the web browser and navigate to http://localhost:5000 (or the provided local server URL).

4. Use the navigation options to explore the app and its features.



   
   
