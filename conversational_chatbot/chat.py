from config import GOOGLE_API_KEY, MODEL_NAME
import google.generativeai as genai

# -----------------------------------------
# Gemini Chat LLM Wrapper
# -----------------------------------------
class GeminiChatLLM:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.chat = self.model.start_chat(history=[])
    
    def __call__(self, prompt, stop=None):
        response = self.chat.send_message(prompt)
        if stop:
            response = response.text.split(stop)[0]
        return response.text.strip()  # Added strip() to clean up the response