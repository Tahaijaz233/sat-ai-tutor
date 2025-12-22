import streamlit as st
from groq import Groq
import os
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from supabase import create_client, Client
import hashlib

# --- 1. CORE SETUP & SUPABASE ---
st.set_page_config(page_title="SAT AI Super Tutor", page_icon="🦉", layout="wide")

try:
    # API Keys
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    # Supabase Credentials
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Missing Secrets in Streamlit!")
    st.stop()

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# AI Engine
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = LlamaGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# --- 2. AUTHENTICATION HELPERS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_user_progress():
    if st.session_state.logged_in:
        supabase.table("profiles").update({
            "score": st.session_state.score,
            "flashcards": st.session_state.flashcards,
            "quests": st.session_state.quests,
            "messages": st.session_state.messages[-15:] # Save last 15 messages
        }).eq("username", st.session_state.username).execute()

# --- 3. LOGIN / SIGNUP SCREEN ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.title("🦉 SAT Super Tutor")
        auth_mode = st.tabs(["Sign In", "Sign Up"])
        
        with auth_mode[0]: # Login
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    res = supabase.table("profiles").select("*").eq("username", u).execute()
                    if res.data and res.data[0]["password_hash"] == hash_password(p):
                        user_data = res.data[0]
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        st.session_state.score = user_data["score"]
                        st.session_state.flashcards = user_data["flashcards"]
                        st.session_state.quests = user_data["quests"]
                        st.session_state.messages = user_data["messages"]
                        st.rerun()
                    else: st.error("Wrong username or password")

        with auth_mode[1]: # Sign Up
            with st.form("signup"):
                new_u = st.text_input("Create Username")
                new_p = st.text_input("Create Password", type="password")
                if st.form_submit_button("Join the Owl"):
                    existing = supabase.table("profiles").select("username").eq("username", new_u).execute()
                    if existing.data: st.error("Username already taken!")
                    else:
                        supabase.table("profiles").insert({
                            "username": new_u, 
                            "password_hash": hash_password(new_p),
                            "quests": {"Ask a question": False, "Use the Camera": False, "Save a Flashcard": False}
                        }).execute()
                        st.success("Account created! Go to 'Sign In'")
    st.stop()

# --- 4. DATA LOADING (RAG) ---
@st.cache_resource(show_spinner=False)
def load_index():
    if not os.path.exists("./data") or not os.listdir("./data"): return None
    return VectorStoreIndex.from_documents(SimpleDirectoryReader("./data").load_data())

sat_index = load_index()

# --- 5. SIDEBAR (Analytics) ---
with st.sidebar:
    st.title(f"👤 {st.session_state.username}")
    if st.button("Logout"):
        save_user_progress()
        st.session_state.logged_in = False
        st.rerun()
    
    # SMART SCORE PREDICTOR
    st.subheader("📈 SAT Analytics")
    pts = st.session_state.score * 5
    st.metric("XP Points", f"{st.session_state.score}", delta=f"+{pts} Est. Points")
    
    # DAILY QUESTS
    st.subheader("🎯 Daily Quests")
    for q, done in st.session_state.quests.items():
        st.checkbox(q, value=done, disabled=True)

    # FLASHCARDS
    st.divider()
    st.subheader("🗂️ My Flashcards")
    for i, card in enumerate(st.session_state.flashcards):
        with st.expander(f"📌 {card['term']}"):
            st.write(card['def'])

# --- 6. CHAT & ACTIONS ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# FLOATING ACTION BAR
st.write("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("📸 Scan Image"): st.session_state.show_camera = True
with c2:
    if st.button("🃏 + Flashcard"): st.session_state.show_flash_ui = True
with c3:
    if st.button("👶 ELI5 Mode"): st.session_state.last_request = "ELI5"
with c4:
    if st.button("💡 Mnemonic"): st.session_state.last_request = "MNEMONIC"

# Feature Modals
if st.session_state.get("show_camera"):
    img = st.camera_input("Snap a problem")
    if img:
        st.session_state.quests["Use the Camera"] = True
        res = genai.GenerativeModel('gemini-2.0-flash').generate_content(["Extract SAT text.", {"mime_type": "image/jpeg", "data": img.getvalue()}])
        st.session_state.messages.append({"role": "user", "content": f"[Scanned Image]: {res.text}"})
        st.session_state.show_camera = False
        save_user_progress()
        st.rerun()

if st.session_state.get("show_flash_ui"):
    with st.form("flash_form"):
        t, d = st.text_input("Term"), st.text_area("Definition")
        if st.form_submit_button("Save"):
            st.session_state.flashcards.append({"term": t, "def": d})
            st.session_state.quests["Save a Flashcard"] = True
            st.session_state.show_flash_ui = False
            save_user_progress()
            st.rerun()

# --- 7. PINNED INPUT ---
if prompt := st.chat_input("Ask your SAT question..."):
    st.session_state.quests["Ask a question"] = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # RAG
    context = ""
    if sat_index:
        nodes = sat_index.as_retriever(similarity_top_k=2).retrieve(prompt)
        context = "\n".join([n.get_text() for n in nodes])

    with st.chat_message("assistant"):
        instr = "Explain like a child." if st.session_state.get("last_request") == "ELI5" else ""
        st.session_state.last_request = None
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": f"Socratic SAT Tutor. Context: {context}. {instr}"}] + st.session_state.messages
        ).choices[0].message.content
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        if "spot on" in response.lower(): st.session_state.score += 2
        
    save_user_progress()
    st.rerun()
