import os
import streamlit as st
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import LLMResult
from pydantic import PrivateAttr
from typing import List, Optional, Any
import base64
import pickle
import hashlib


# --- Page Config ---
st.set_page_config(page_title="☁️ AI Marketing Generator", layout="centered")

# --- Constants ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
USERS_FILE = "users.pkl"
CACHE_FILE = "generation_cache.pkl"
BACKGROUND_IMAGE = "background_image.jpg"

# --- Load Users ---
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "rb") as f:
        users = pickle.load(f)
else:
    users = {
        "hari@gmail.com": {"password": hashlib.sha256("admin123".encode()).hexdigest()}
    }

def save_users():
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Gemini Wrapper ---
class GeminiLLM(LLM):
    model_name: str = "models/gemini-1.5-flash"
    _model: Any = PrivateAttr()

    def __init__(self, api_key: str, model_name: str = "models/gemini-1.5-flash"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name=model_name)

    @property
    def _llm_type(self) -> str:
        return "google-gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate_content(prompt)
        return response.text.strip()

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> LLMResult:
        generations = [[{"text": self._call(prompt, stop)}] for prompt in prompts]
        return LLMResult(generations=generations)

# --- Background Styling ---
def get_base64_bg(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = get_base64_bg(BACKGROUND_IMAGE)
