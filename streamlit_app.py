import streamlit as st
import requests
from bs4 import BeautifulSoup
import ollama

MODEL = "llama3.2"

class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = "You are an assistant that analyzes the contents of a website and provides a short summary, ignoring text that might be navigation related. Respond in markdown."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "The contents of this website is as follows; please provide a short summary of this website in markdown. If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    messages = messages_for(website)
    response = ollama.chat(model=MODEL, messages=messages)
    return response['message']['content']

def display_summary(url):
    summary = summarize(url)
    st.markdown(summary)

def transcript_chat_completion_ollama(transcript, user_question, model="llama3.2"):
    prompt = f"""Use this transcript or transcripts to answer any user questions, citing specific quotes:
    {transcript}
    Question: {user_question}
    Answer:
    """
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        st.error(f"Error during Ollama generation: {e}")
        return "Error processing the request."

st.title("Website Analyzer")

url = st.text_input("Enter the website URL:")
if url:
    st.subheader("Website Summary")
    display_summary(url)

    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question:")
    if user_question:
        transcript = summarize(url)
        answer = transcript_chat_completion_ollama(transcript, user_question, model="llama3.2")
        st.write(answer)
