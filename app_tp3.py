import os
import json
import requests
import streamlit as st
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="مبارك ولد حميدة — Agent TP3",
    page_icon="🍵",
    layout="wide"
)

# ── CSS (RTL, dark tea-merchant theme) ───────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&display=swap');
  html, body, [class*="css"] { font-family: 'Amiri', serif; background-color: #1a0a00; color: #f5deb3; }
  .stApp { background-color: #1a0a00; }
  .main-header { text-align: center; color: #d4a843; font-size: 2.4rem; font-weight: bold; margin-bottom: 0; }
  .sub-header  { text-align: center; color: #a0784a; font-size: 1rem; margin-bottom: 1.5rem; }
  .user-bubble {
    background: #2d1a0a; border-left: 3px solid #a0784a;
    padding: 10px 14px; border-radius: 8px; margin: 8px 0;
    text-align: right; direction: rtl; color: #f5deb3;
  }
  .bot-bubble {
    background: #3d2010; border-left: 3px solid #d4a843;
    padding: 10px 14px; border-radius: 8px; margin: 8px 0;
    text-align: right; direction: rtl; color: #fff8e1;
  }
  .meta-badge {
    display: inline-block; background: #4a2a10; color: #d4a843;
    border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; margin: 2px;
  }
  .confidence-forte   { color: #4caf50; font-weight: bold; }
  .confidence-moyenne { color: #ff9800; font-weight: bold; }
  .confidence-faible  { color: #f44336; font-weight: bold; }
  .stTextInput > div > div > input { direction: rtl; text-align: right; background: #2d1a0a; color: #f5deb3; border: 1px solid #a0784a; }
  .stButton > button { background: #8b4513; color: #f5deb3; border: 1px solid #d4a843; }
  .stButton > button:hover { background: #d4a843; color: #1a0a00; }
  .sidebar-title { color: #d4a843; font-size: 1.1rem; font-weight: bold; }
  div[data-testid="stSidebar"] { background-color: #110600; }
</style>
""", unsafe_allow_html=True)

# ── Character profile ─────────────────────────────────────────────────────────
CHARACTER_PROFILE = """
Tu es مبارك ولد حميدة (Mbark ould Hmeyda), vieux marchand de thé (أتاي),
de dattes (التمر) et de warka (وركة) au marché central de Nouakchott, Mauritanie.

Langue : Hassaniya (dialecte arabe mauritanien), avec quelques mots arabes classiques.
Ton : chaleureux, sage, paternel, ancré dans la foi islamique.
Expressions : "هيه", "الله يبارك فيك", "إن شاء الله", "ربي يحفظك", "الزبون سيدك".
Règles :
  1. Réponds TOUJOURS en Hassaniya en priorité.
  2. Note courte en français entre parenthèses pour les infos factuelles.
  3. Cite la source et le niveau de confiance à la fin.
  4. Si tu ne sais pas : "والله ما عندي خبر".
  5. Ne jamais inventer un prix, une heure ou un lieu.
"""

# ── Proverbs (fallback si dataset HF non disponible) ─────────────────────────
PROVERBES = [
    {"hassaniya": "الصبر مفتاح الفرج",       "fr": "La patience est la clé du soulagement",      "themes": ["patience", "conseil"]},
    {"hassaniya": "الزبون سيدك",              "fr": "Le client est ton maître",                   "themes": ["commerce", "vente"]},
    {"hassaniya": "البركة في البكور",          "fr": "La bénédiction est dans le lever tôt",       "themes": ["travail", "matin"]},
    {"hassaniya": "اللي يشرب أتاي بصبر يفهم طعمو", "fr": "Patience révèle le goût du thé",       "themes": ["thé", "patience"]},
    {"hassaniya": "الحق يغلب",                "fr": "La vérité triomphe",                         "themes": ["vérité", "honnêteté"]},
    {"hassaniya": "العلم نور",                "fr": "Le savoir est une lumière",                  "themes": ["savoir", "éducation"]},
    {"hassaniya": "الوقت من ذهب",             "fr": "Le temps c'est de l'or",                    "themes": ["temps", "travail"]},
    {"hassaniya": "اللي عنده أتاي عنده ضيف",  "fr": "Le thé attire toujours un hôte",            "themes": ["hospitalité", "thé"]},
]

# ── Tool functions ────────────────────────────────────────────────────────────
def _get_weather(city: str = "Nouakchott") -> dict:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=8
        ).json()
        if not geo.get("results"):
            return {"erreur": f"Ville inconnue: {city}"}
        loc = geo["results"][0]
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"], "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                "timezone": "auto"
            }, timeout=8
        ).json()["current"]
        return {
            "ville": city, "temperature_c": w["temperature_2m"],
            "humidite_pct": w["relative_humidity_2m"],
            "vent_kmh": w["wind_speed_10m"],
            "source": "Open-Meteo API"
        }
    except Exception as e:
        return {"erreur": str(e)}

def _get_prayer_times(city: str = "Nouakchott", country: str = "Mauritania") -> dict:
    try:
        t = requests.get(
            "https://api.aladhan.com/v1/timingsByCity",
            params={"city": city, "country": country, "method": 3}, timeout=8
        ).json()["data"]["timings"]
        return {
            "Fajr": t["Fajr"], "Dhuhr": t["Dhuhr"],
            "Asr": t["Asr"], "Maghrib": t["Maghrib"], "Isha": t["Isha"],
            "source": "AlAdhan Prayer Times API"
        }
    except Exception as e:
        return {"erreur": str(e)}

def _get_exchange_rate(from_cur: str = "EUR", to_cur: str = "USD", amount: float = 1.0) -> dict:
    try:
        data = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": from_cur, "to": to_cur, "amount": amount}, timeout=8
        ).json()
        rate = data["rates"].get(to_cur)
        if rate is None:
            return {"erreur": f"Devise non supportée: {to_cur}"}
        return {
            "de": f"{amount} {from_cur}", "vers": f"{rate:.4f} {to_cur}",
            "date": data["date"], "source": "Frankfurter Exchange Rate API"
        }
    except Exception as e:
        return {"erreur": str(e)}

def _search_proverbs(query: str) -> list:
    q = query.lower()
    scored = []
    for p in PROVERBES:
        score = sum(2 for t in p["themes"] if t in q)
        score += sum(1 for w in q.split() if w in p["hassaniya"] + p["fr"].lower())
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        import random
        return random.sample(PROVERBES, 2)
    return [p for _, p in scored[:3]]

def _geocode(place: str) -> dict:
    try:
        headers = {"User-Agent": "tp3-mbark-app/1.0 (educational)"}
        for q in [f"{place} Mauritania", place]:
            data = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "json", "limit": 1, "accept-language": "fr"},
                headers=headers, timeout=8
            ).json()
            if data:
                loc = data[0]
                return {
                    "lieu": place,
                    "nom_complet": loc.get("display_name", ""),
                    "latitude": loc.get("lat"), "longitude": loc.get("lon"),
                    "source": "OpenStreetMap / Nominatim"
                }
        return {"erreur": f"Lieu introuvable: {place}"}
    except Exception as e:
        return {"erreur": str(e)}

# ── LangChain agent (cached per API key) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_agent(groq_key: str):
    os.environ["GROQ_API_KEY"] = groq_key
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str = "Nouakchott") -> dict:
        """Météo actuelle d'une ville : température, vent, humidité. Pour conseiller sur les conditions de marché."""
        return _get_weather(city)

    @tool
    def get_prayer_times(city: str = "Nouakchott", country: str = "Mauritania") -> dict:
        """Horaires des 5 prières quotidiennes selon ville et pays. Essentiel pour un marchand musulman."""
        return _get_prayer_times(city, country)

    @tool
    def get_exchange_rate(from_currency: str = "EUR", to_currency: str = "USD", amount: float = 1.0) -> dict:
        """Convertir des devises (EUR, USD, GBP, MAD...). Utile pour les échanges commerciaux internationaux."""
        return _get_exchange_rate(from_currency, to_currency, amount)

    @tool
    def search_proverbs(query: str) -> list:
        """Proverbes hassaniya sur un thème : patience, commerce, thé, temps, sagesse, famille."""
        return _search_proverbs(query)

    @tool
    def geocode_place(place_name: str) -> dict:
        """Coordonnées et infos d'un lieu en Mauritanie ou ailleurs (marché, quartier, ville)."""
        return _geocode(place_name)

    # parallel_tool_calls=False évite le bug texte+tool_call dans le même tour
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        model_kwargs={"parallel_tool_calls": False}
    )
    tools = [get_weather, get_prayer_times, get_exchange_rate, search_proverbs, geocode_place]
    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=llm, tools=tools,
        checkpointer=checkpointer,
        prompt=CHARACTER_PROFILE
    )
    return agent, tools

def invoke_agent(agent, tools, user_input: str, thread_id: str) -> dict:
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    messages = result["messages"]
    final = messages[-1].content
    tools_called, sources = [], []
    tool_names = [t.name for t in tools]
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
        if hasattr(msg, "name") and msg.name in tool_names:
            try:
                content = json.loads(msg.content) if isinstance(msg.content, str) else {}
                sources.append(content.get("source", msg.name))
            except Exception:
                sources.append(msg.name)
    return {
        "response": final,
        "action": tools_called[0].upper() if tools_called else "DIRECT_RESPONSE",
        "tools_used": tools_called,
        "sources": sources or ["Connaissance du personnage (TP2)"],
        "confidence": "forte" if tools_called else "moyenne"
    }

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S')}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ Configuration</p>', unsafe_allow_html=True)
    openai_key = st.text_input("🔑 OpenAI API Key", type="password",
                                help="Entrez votre clé OpenAI pour activer l'agent")
    st.divider()
    st.markdown('<p class="sidebar-title">💬 Session</p>', unsafe_allow_html=True)
    st.code(st.session_state.thread_id, language=None)
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.rerun()
    st.divider()
    st.markdown('<p class="sidebar-title">🔧 Outils disponibles</p>', unsafe_allow_html=True)
    st.markdown("""
- 🌤 Météo (Open-Meteo)
- 🕌 Prières (AlAdhan)
- 💱 Change (Frankfurter)
- 📜 Proverbes Hassaniya
- 📍 Géocodage (OSM)
    """)
    st.divider()
    st.markdown('<p class="sidebar-title">📊 LangSmith</p>', unsafe_allow_html=True)
    langsmith_key = st.text_input("🔑 LangSmith API Key", type="password")
    if langsmith_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "tp3-mbark-hassaniya-agent"
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        st.success("Tracing activé")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">مبارك ولد حميدة</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">تاجر يبيع الأتاي  في نواكشوط — Agent LangChain TP3</p>', unsafe_allow_html=True)
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">🍵 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if "meta" in msg:
                m = msg["meta"]
                conf_class = f"confidence-{m.get('confidence', 'moyenne')}"
                st.markdown(
                    f'<span class="meta-badge">⚡ {m.get("action","—")}</span>'
                    f'<span class="meta-badge">📍 {" | ".join(m.get("sources",[]))}</span>'
                    f'<span class="meta-badge {conf_class}">💪 {m.get("confidence","—")}</span>',
                    unsafe_allow_html=True
                )

# ── Input ─────────────────────────────────────────────────────────────────────
st.divider()
user_input = st.chat_input("سول مبارك... (posez votre question en Hassaniya ou en français)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not openai_key:
        st.warning("Entrez votre OpenAI API Key dans la barre latérale pour activer l'agent.")
    else:
        with st.spinner("مبارك يتخمم..."):
            try:
                agent, tools = build_agent(openai_key)
                result = invoke_agent(agent, tools, user_input, st.session_state.thread_id)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "meta": {
                        "action":     result["action"],
                        "sources":    result["sources"],
                        "confidence": result["confidence"],
                        "thread_id":  st.session_state.thread_id
                    }
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"والله ما نعرف — Erreur : {e}"
                })
    st.rerun()
