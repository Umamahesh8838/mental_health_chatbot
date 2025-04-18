import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import google.generativeai as genai

genai.configure(api_key="AIzaSyCAmwoJ7AjyOQB-7W4GqpW78Wli5d5aeis")  # Replace with your Gemini key
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

emotion_classifier = load_emotion_model()

def predict_emotion(text):
    result = emotion_classifier(text)
    emotion = result[0]['label']
    confidence = round(result[0]['score'], 2)
    return emotion, confidence

st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")
st.title("🧠 Emotion-Aware Mental Health Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:")

if user_input:
    emotion, confidence = predict_emotion(user_input)
    st.session_state.chat_history += f"\nUser ({emotion}): {user_input}"
    prompt = (
        "You are a kind and emotionally intelligent chatbot.\n"
        f"Conversation so far:\n{st.session_state.chat_history}\n\n"
        f"Respond empathetically to the user's message (emotion: {emotion}):"
    )
    response = gemini_model.generate_content(prompt).text.strip()
    st.session_state.chat_history += f"\nBot: {response}"
    st.session_state.messages.append((user_input, emotion, confidence, response))

for u, e, c, b in st.session_state.messages:
    st.markdown(f"**You** ({e}, confidence: {c}): {u}")
    st.markdown(f"**Chatbot** 🤖: {b}")
    st.markdown("---")

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = ""
    st.session_state.messages = []
