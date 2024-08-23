import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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

# Step 4: Apply custom CSS for better UI
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
            color: #ffffff;  /* Set the color to white for better readability */
            margin: 5px 0;
        }
        .separator {
            border: none;
            border-top: 1px solid #e6e6e6;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Step 5: Streamlit Interface with Conversation Loop
def main():
    apply_custom_css()

    st.markdown("<h1 class='stTitle'>Twitter Q&A Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='stText'>Ask me anything about Twitter!</p>", unsafe_allow_html=True)

    # Initialize session state to store conversation history and exit flag
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'exit' not in st.session_state:
        st.session_state.exit = False

    # If the exit flag is True, show the goodbye message and stop further input
    if st.session_state.exit:
        st.markdown("<p class='stText'>Goodbye! See you next time.</p>", unsafe_allow_html=True)
    else:
        # Handle the user input via text box
        with st.form(key='input_form'):
            user_input = st.text_input("You: ", value="", key="input_box")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            if user_input.lower() == "exit":
                st.session_state.exit = True  # Set the exit flag to True
                st.experimental_rerun()  # Rerun the app to update the interface
            else:
                matched_question, answer = chatbot_response(user_input)
                st.session_state.conversation.append((user_input, matched_question, answer))
                st.experimental_rerun()  # Rerun the app to clear the input box

        # Display the conversation history
        for i, (user_q, matched_q, bot_a) in enumerate(st.session_state.conversation):
            st.markdown(f"""
                <div class="conversation">
                    <p><strong>You:</strong> {user_q}</p>
                    <p><strong>Matched Question:</strong> {matched_q}</p>
                    <p><strong>Answer:</strong> {bot_a}</p>
            """, unsafe_allow_html=True)
            st.markdown("</div><hr class='separator'/>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
