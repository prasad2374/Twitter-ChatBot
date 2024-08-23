import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pyttsx3
import speech_recognition as sr
import threading

# Step 1: Load the CSV File
csv_path = 'twitter_questions_and_answers.csv'  # Update with your file path if necessary
df = pd.read_csv(csv_path)

# Step 2: Preprocess the Data using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Question'])

# Step 3: Implement the Chatbot Function
def chatbot_response(user_input):
    # Transform the user input using the same TF-IDF vectorizer
    user_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Calculate the cosine similarity between the user input and all questions in the dataset
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Find the index of the most similar question
    best_match_index = cosine_similarities.argmax()
    
    # Get the most similar question and the corresponding answer
    best_match_question = df['Question'].iloc[best_match_index]
    best_match_answer = df['Answer'].iloc[best_match_index]
    
    return best_match_question, best_match_answer

# Step 4: Microphone Input
def get_audio_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        st.write(f"Recognized: {user_input}")
        return user_input
    except sr.UnknownValueError:
        st.write("Could not understand the audio")
        return None  # Return None instead of an empty string for better handling
    except sr.RequestError:
        st.write("Could not request results; check your network connection")
        return None  # Return None instead of an empty string for better handling

# Step 5: Function to speak the answer in a separate thread
def speak_text(text):
    def speak():
        # Reinitialize the TTS engine for each speech request
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=speak).start()

# Step 6: Apply custom CSS for better UI
def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f4f4f4;
        }
        .stApp {
            background-color: #f4f4f4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stTitle {
            color: #333333;
            font-weight: 600;
            font-size: 36px;
            text-align: center;
        }
        .stText {
            color: #000000;
            font-size: 18px;
        }
        .stButton button {
            background-color: #0073e6;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #cccccc;
            padding: 10px;
            font-size: 16px;
        }
        .stForm {
            margin-bottom: 20px;
        }
        .conversation {
            background-color: #333333;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
        }
        .conversation p {
            color: #000000  /* Set the color to black */
            margin: 5px 0;
        }
        .separator {
            border: none;
            border-top: 1px solid #e6e6e6;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Step 7: Streamlit Interface with Conversation Loop
def main():
    apply_custom_css()

    st.markdown("<h1 class='stTitle'>Twitter Q&A Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='stText'>Ask me anything about Twitter!</p>", unsafe_allow_html=True)

    # Initialize session state to store conversation history, exit flag, and selected answer
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'exit' not in st.session_state:
        st.session_state.exit = False
    if 'selected_answer' not in st.session_state:
        st.session_state.selected_answer = None

    # If the exit flag is True, show the goodbye message and stop further input
    if st.session_state.exit:
        st.markdown("<p class='stText'>Goodbye! See you next time.</p>", unsafe_allow_html=True)
    else:
        # Handle the user input via text box or microphone
        with st.form(key='input_form'):
            user_input = st.text_input("You: ", value="", key="input_box")
            submit_button = st.form_submit_button(label='Send')
        
        mic_button = st.button("Use Microphone")
        
        if mic_button:
            user_input = get_audio_input()

        if (submit_button and user_input) or (mic_button and user_input):
            if user_input.lower() == "exit":
                st.session_state.exit = True  # Set the exit flag to True
                st.experimental_rerun()  # Rerun the app to update the interface
            else:
                matched_question, answer = chatbot_response(user_input)
                st.session_state.conversation.append((user_input, matched_question, answer))
                st.session_state.selected_answer = answer  # Store the selected answer
                st.experimental_rerun()  # Rerun the app to clear the input box

        # Display the conversation history and add a "Speak" button for the last answer
        for i, (user_q, matched_q, bot_a) in enumerate(st.session_state.conversation):
            st.markdown(f"""
                <div class="conversation">
                    <p><strong>You:</strong> {user_q}</p>
                    <p><strong>Matched Question:</strong> {matched_q}</p>
                    <p><strong>Answer:</strong> {bot_a}</p>
            """, unsafe_allow_html=True)
            if i == len(st.session_state.conversation) - 1:  # Only add the button for the latest answer
                if st.button("Speak", key=f"speak_button_{i}"):
                    speak_text(bot_a)
            st.markdown("</div><hr class='separator'/>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
