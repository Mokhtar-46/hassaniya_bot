# app.py — Hassaniya Chatbot (Mbark ould Hmeyda)
# Run: streamlit run app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb, os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="مبارك ولد حميدة",
    page_icon="🍵",
    layout="centered",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Amiri', serif;
    direction: rtl;
}

.stApp {
    background: linear-gradient(160deg, #1a0a00 0%, #2d1200 50%, #1a0800 100%);
    color: #f5deb3;
}

/* Header */
.chat-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid #8b4513;
    margin-bottom: 1.5rem;
}
.chat-header h1 {
    font-size: 2.2rem;
    color: #d4a04a;
    margin: 0;
    text-shadow: 0 2px 8px rgba(212,160,74,0.3);
}
.chat-header p {
    color: #a07040;
    font-size: 1rem;
    margin: 0.3rem 0 0 0;
}

/* Chat bubbles */
.msg-user {
    background: #2d1a00;
    border: 1px solid #8b4513;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #f5deb3;
    font-size: 1.05rem;
    text-align: right;
}
.msg-bot {
    background: #1a0d00;
    border: 1px solid #5c3010;
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #e8c88a;
    font-size: 1.05rem;
    text-align: right;
}
.msg-label-user { color: #a07040; font-size: 0.8rem; text-align: left; margin-bottom: 2px; }
.msg-label-bot  { color: #6b4020; font-size: 0.8rem; text-align: right; margin-bottom: 2px; }

/* Input */
.stTextInput > div > div > input {
    background: #2d1200 !important;
    color: #f5deb3 !important;
    border: 1px solid #8b4513 !important;
    border-radius: 12px !important;
    font-family: 'Amiri', serif !important;
    font-size: 1.05rem !important;
    direction: rtl !important;
    text-align: right !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #8b4513, #d4a04a) !important;
    color: #1a0800 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Amiri', serif !important;
    font-weight: bold !important;
    font-size: 1rem !important;
    padding: 0.5rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #d4a04a, #8b4513) !important;
    transform: translateY(-1px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #150800 !important;
    border-right: 1px solid #5c3010;
}
section[data-testid="stSidebar"] * {
    color: #c4904a !important;
    direction: rtl;
}

/* Scrollable chat area */
.chat-container {
    max-height: 420px;
    overflow-y: auto;
    padding: 0.5rem;
    margin-bottom: 1rem;
    scrollbar-width: thin;
    scrollbar-color: #8b4513 #1a0800;
}

.tea-divider {
    text-align: center;
    color: #5c3010;
    font-size: 1.2rem;
    margin: 0.5rem 0;
    letter-spacing: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "./mbark_model")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model — from local path or download from W&B."""

    # Try local first
    if os.path.exists(MODEL_PATH) and os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
        return tokenizer, model

    # Download from W&B artifact
    wandb_key = "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"
    if not wandb_key:
        st.error("⚠️ MODEL_PATH not found and WANDB_API_KEY not set. See sidebar.")
        st.stop()

    wandb.login(key=wandb_key)
    api      = wandb.Api()
    artifact = api.artifact("oussallay200-esp/hassaniya-mbark/mbark-hassaniya-model:latest")
    artifact.download(root=MODEL_PATH)
    wandb.finish()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    return tokenizer, model


def generate_response(tokenizer, model, question,
                      max_new_tokens=80, temperature=0.8,
                      top_p=0.9, rep_penalty=1.3):
    prompt     = f"سؤال: {question}\nجواب: "
    inputs     = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = temperature,
            top_p              = top_p,
            repetition_penalty = rep_penalty,
            pad_token_id       = tokenizer.eos_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    temperature  = st.slider("درجة الإبداع", 0.5, 1.2, 0.8, 0.05)
    rep_penalty  = st.slider("عقوبة التكرار", 1.0, 2.0, 1.3, 0.1)
    max_tokens   = st.slider("أقصى طول للجواب", 30, 150, 80, 10)
    st.markdown("---")
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**مبارك ولد حميدة**")
    st.markdown("تاجر شاي وتمر قديم من نواكشوط")
    st.markdown("يتكلم الحسانية")

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="chat-header">
    <h1>🍵 مبارك ولد حميدة</h1>
    <p>تاجر يبيع الوركة  — نواكشوط</p>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "السلام عليكم! أنا مبارك، كيف   انت شدور 🍵"}
    ]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

with st.spinner("⏳ جاري تحميل النموذج..."):
    tokenizer, model = load_model()

# ── CHAT DISPLAY ──────────────────────────────────────────────────────────────

chat_html = '<div class="chat-container">'
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f'<div class="msg-label-user">أنت</div>'
        chat_html += f'<div class="msg-user">{msg["text"]}</div>'
    else:
        chat_html += f'<div class="msg-label-bot">مبارك 🍵</div>'
        chat_html += f'<div class="msg-bot">{msg["text"]}</div>'
chat_html += '</div>'

st.markdown(chat_html, unsafe_allow_html=True)
st.markdown('<div class="tea-divider">🍵 ✦ 🍵</div>', unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label     = "رسالتك",
        placeholder = "اكتب سؤالك هنون...",
        label_visibility = "collapsed",
        key       = "input_box",
    )
with col2:
    send = st.button("إرسال")

if send and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "text": question})

    with st.spinner("مبارك يتخمم... 🍵"):
        answer = generate_response(
            tokenizer, model, question,
            max_new_tokens = max_tokens,
            temperature    = temperature,
            rep_penalty    = rep_penalty,
        )

    if not answer:
        answer = "الله يسهل، ما فهمتش مليح، عاود قول"

    st.session_state.messages.append({"role": "bot", "text": answer})
    st.rerun()
