# Standard library imports
import os
import logging

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
from chat import GeminiChatLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Local application/library imports
from logging_config import *
from faiss_index import GeminiEmbeddings
from config import GOOGLE_API_KEY, MODEL_NAME

logging.info("App started.")

# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it in your .env file.")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME is not set. Please set it in your .env file.")


# -----------------------------------------
# Retrieval + Chat with Memory
# -----------------------------------------
def query(question, chat_history):
    try:
        logging.info("Processing query...")
        gemini = GeminiEmbeddings(api_key=GOOGLE_API_KEY)
        db = FAISS.load_local("faiss_index", embeddings=gemini, allow_dangerous_deserialization=True)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY),
            retriever=db.as_retriever(),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        # Build manual context from chat history
        context = ""
        for prev_q, prev_a in chat_history:
            context += f"User: {prev_q}\nAI: {prev_a}\n"
        context += f"User: {question}\nAI:"

        # Feed full context to Gemini
        response = chain.invoke({"question": context, "chat_history": chat_history})
        logging.info("Query processed successfully.")
        # Extract answer from response
        answer = response["answer"]
        return answer
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        raise

# -----------------------------------------
# Streamlit Chat UI
# -----------------------------------------
def show_ui():
    st.set_page_config(page_title="HR Policy Chatbot", layout="wide")
    st.title("🤖 Yours Truly Human Resources Chatbot")
    st.subheader("Ask anything about your company's HR policies")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter your HR query here..."):
        with st.spinner("🔍 Searching HR policies..."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(response)

            # Save to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append((prompt, response))

# -----------------------------------------
# Entry Point
# -----------------------------------------
if __name__ == "__main__":
    show_ui()
