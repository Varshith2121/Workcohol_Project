# ☁️ AI Marketing Idea Generator

This is a Streamlit-based web application that uses Google Gemini (via LangChain) to generate creative marketing slogans, ad copies, and campaign ideas based on user input. It is a streamlined web app that helps startups, businesses, and marketers create high-converting slogans, ad copy, and campaign ideas in seconds. Powered by LangChain for advanced prompt orchestration and Streamlit for a fast, intuitive UI, this tool puts AI-driven creativity at your fingertips.

![Image of the Interface](pexels-leofallflat-1737957.jpg)

---

 Features

- Choose between Slogan, Ad Copy, and Campaign Idea generation.
- Simply enter a brief description about Brand/Product- the AI takes care of the rest.
- Stylish UI with custom background image and dark blur theme.
- Separate login/logout/dashboard views with session handling.
- Pickle is used to store user data and cache generated results.
- Uses Google Gemini via LangChain's custom LLM wrapper.
- Get results in formats suitable for social media, email marketing, web banners, and more.

---

 Technologies Used

- Python
- Streamlit
- Google Generative AI (Gemini)
- Pickle
- Hashlib
- Pydantic
- LangChain
- Streamlit for API keys
- Base64/CSS for background styling

---

  How It Works

1. API Key is loaded from Streamlit secrets.
2. A custom wrapper around Gemini is built by extending `langchain.llms.base.LLM`.
3. Prompt templates are selected dynamically based on user choice.
4. LangChain's `LLMChain` is used to call Gemini and return results.
5. Typing animation & styled output enhances user experience.

---

  🌐 Live Demo

 You can try the deployed app here: [AI Marketing Idea Generator](https://aimarketgenerator.streamlit.app/)
