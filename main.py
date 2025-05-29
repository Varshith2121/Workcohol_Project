import os
import streamlit as st
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import LLMResult
from pydantic import PrivateAttr
from typing import List, Optional, Any
import time
import base64
import pickle

# --- Streamlit page config ---
st.set_page_config(page_title="☁️ AI Marketing Generator", layout="centered")

# --- Load API Key from Streamlit Secrets ---
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("API key not found in secrets.toml. Add GEMINI_API_KEY.")
    st.stop()

# --- Helper to embed background image ---
def get_base64_bg(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_path = "pexels-leofallflat-1737957.jpg"
bg_base64 = get_base64_bg(background_path)

# --- Custom CSS Styling ---
st.markdown(f"""<style>
html, body, [data-testid="stApp"] {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: #f3f3f3;
    font-family: 'Segoe UI', sans-serif;
}}
... (CSS remains unchanged) ...
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="title-container">
    <h1>☁️ AI Marketing Idea Generator</h1>
    <p class="subtitle">Generate catchy <b>slogans</b>, <b>ad copy</b> or <b>campaign ideas.</b></p>
</div>
""", unsafe_allow_html=True)

# --- Gemini LLM Wrapper ---
class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    _model: Any = PrivateAttr()

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
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

# --- Typewriter Effect ---
def type_writer_effect(text, speed=0.02):
    output = ""
    placeholder = st.empty()
    for char in text:
        output += char
        placeholder.markdown(f"<div class='typing'>{output}</div>", unsafe_allow_html=True)
        time.sleep(speed)

# --- Caching ---
CACHE_FILE = "generation_cache.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

# --- Main App Logic ---
task_type = st.selectbox(" Select what to generate:", ["Slogan", "Ad Copy", "Campaign Idea"])
user_input = st.text_input(" Describe your product or brand:", "e.g. 'A new eco-friendly water bottle'").strip()

prompt_templates = {
    "Slogan": "Create a catchy marketing slogan for: {product}",
    "Ad Copy": "Write a persuasive ad copy for: {product}",
    "Campaign Idea": "Come up with a creative marketing campaign for: {product}"
}

if user_input:
    if st.button("🚀 Generate"):
        key = (task_type, user_input)
        if key in cache:
            result = cache[key]
            st.success("Loaded from cache!")
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = GeminiLLM(api_key=API_KEY)
                    prompt = PromptTemplate.from_template(prompt_templates[task_type])
                    chain = LLMChain(llm=llm, prompt=prompt)
                    result = chain.run(product=user_input)
                    cache[key] = result
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(cache, f)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    result = None

        if result:
            st.markdown(" 🎯 Generated Output")
            st.markdown(f'<div class="output-box" id="output-text">{result}</div>', unsafe_allow_html=True)
            st.markdown("""
                <button class="copy-button" onclick="navigator.clipboard.writeText(document.getElementById('output-text').innerText);alert('Copied to clipboard!');">
                    📋 Copy to Clipboard
                </button>
                """, unsafe_allow_html=True)
else:
    st.info("Fill in the product/brand description to begin.")
