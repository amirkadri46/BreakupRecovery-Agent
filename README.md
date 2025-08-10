## 💔 Breakup Recovery Squad – AI Support App

A streamlined Streamlit app that gives compassionate, practical support after a breakup. Your personal recovery squad features four specialized AI friends who respond with empathy, truth, closure guidance, and step‑by‑step momentum.

### Why this exists
Heartbreak is heavy. This app makes it easier to talk, reflect, and get grounded guidance—fast. You can chat one‑on‑one with a specific agent or hear from the whole squad.

---

## ✨ Features
- **Recovery squad**: Four focused agents with distinct roles
  - **Maya (💙)**: empathy and emotional grounding
  - **Alex (✨)**: closure and letting go
  - **Jordan (🚀)**: plans, routines, and momentum
  - **Sam (💪)**: honest reality checks with care
- **Group or 1‑on‑1**: Choose a single agent or “All Squad Members”.
- **Rich context**: Upload screenshots or documents (`.png`, `.jpg`, `.jpeg`, `.txt`, `.docx`, `.pdf`).
- **Initial support**: One click to get a comprehensive first response from the whole squad.
- **Chat UI**: Continue the conversation in a clean, threaded interface.
- **Friendly error handling**: Clear messages when rate‑limited or misconfigured.

Note: The optional “Search Resources” sidebar was removed for a cleaner experience.

---

## 🧰 Tech Stack
- **Frontend**: `streamlit`
- **AI Model**: Google Gemini (via `google-genai`)
- **Agent Framework**: `agno`
- **Docs**: `python-docx`, `pypdf`

---

## 🚀 Quickstart

### 1) Prerequisites
- Python 3.10+ (3.11 recommended)
- A Google AI Studio API key for Gemini

Get an API key:
- Create a key at [Google AI Studio](https://makersuite.google.com/app/apikey)
- Ensure the Generative Language API is enabled in your Google Cloud project: [Enable API](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com)

### 2) Clone and install
```bash
git clone "<your-repo-url>"
cd "breakup recovery AI-agent"
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run breakup_agent.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

---

## 🔐 API Key
- Enter your `GEMINI_API_KEY` in the sidebar field labeled “Enter your GEMINI_API_KEY”.
- The app does not currently read an environment variable; the sidebar field is required.

---

## 🧭 Using the App
1. **Enter your API key** in the sidebar. Wait for “Recovery squad ready!”.
2. **Choose who to chat with** in `💬 Chat Settings`:
   - `All Squad Members` for a group response
   - Or pick an individual agent for focused support
3. **Share your story** in the main input area. Optionally **upload screenshots/docs** for context. PDFs are limited to 5 pages; long text is trimmed to keep responses focused.
4. Click **“Get Initial Support from Your Squad 💝”** for a multi‑agent first response.
5. Use the **chat box** at the bottom to continue the conversation.

Tips:
- Short, clear messages lead to the most helpful replies.
- When uploading files, include the parts that matter most.

---

## 🧩 Project Structure
```
breakup recovery AI-agent/
├─ breakup_agent.py        # Streamlit app entry & UI
├─ requirements.txt        # Python dependencies
└─ README.md               # You are here
```

Key components in `breakup_agent.py`:
- `RecoverySquadApp` – the main app class
- `initialize_agents` – builds agents (Maya, Alex, Jordan, Sam)
- `_render_agent_selection`, `render_main_content`, `render_chat_section` – UI
- `_provide_initial_support`, `get_agent_response` – core interaction logic

---

## 🧪 Troubleshooting
- **“Please provide the API key to proceed”**: Enter your key in the sidebar.
- **“Initializing your recovery squad…” never finishes**: Check your internet connection and that the key is valid.
- **429 / “Too Many Requests”**: The model is rate‑limited. Try again after a short wait, or chat with one agent at a time.
- **PDF/text not fully processed**: The app intentionally limits PDF pages (5) and total extracted text (about 4000 chars) to keep responses fast and focused.
- **Blank page or Streamlit error**: Stop the app and rerun `streamlit run breakup_agent.py`.

---

## 🛠️ Development
- Keep functions small and well‑named; prefer early returns.
- Match the existing formatting style.
- Run the app locally with `streamlit run breakup_agent.py` and watch the console for errors.

### Useful commands
```bash
pip install -r requirements.txt
pip freeze > requirements.txt
streamlit run breakup_agent.py
```

---

## 🤝 Contributing
Issues and suggestions are welcome. Please include:
- What you expected vs. what happened
- Steps to reproduce
- Screenshots or sample text (if possible)

---

## 🙏 A gentle note
This app is for emotional support and guidance. It is not a substitute for professional mental health care. If you’re in crisis, please reach out to local emergency services or a trusted professional.


