# ☁️ AI Marketing Idea Generator

This is a Streamlit-based web application that uses Google Gemini (via LangChain) to generate creative marketing slogans, ad copies, and campaign ideas based on user input.

![Image of the Interface](image.png)

---

 Features

- Choose between Slogan, Ad Copy, and Campaign Idea generation.
- Input your brand or product description** and get instant results.
- Built-in copy to clipboard functionality.
- Stylish UI with custom background image and dark blur theme.
- Uses Google Gemini via LangChain's custom LLM wrapper.

---

 Technologies Used

- Python
- Streamlit
- Google Generative AI (Gemini)
- LangChain
- .env for API keys
- Base64/CSS for background styling

---

  How It Works

1. API Key is loaded from `.env` or Streamlit secrets.
2. A custom wrapper around Gemini is built by extending `langchain.llms.base.LLM`.
3. Prompt templates are selected dynamically based on user choice.
4. LangChain's `LLMChain` is used to call Gemini and return results.
5. Typing animation & styled output enhances user experience.

---

