import streamlit as st
from groq import Groq
import os
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core import Settings

# 1. THE SETUP
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # For the Camera
client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"
Settings.llm = LlamaGroq(model=MODEL_NAME, api_key=GROQ_API_KEY)

# 2. CONFIGURE FREE MODELS
genai.configure(api_key=GOOGLE_API_KEY)

# 3. THE LIBRARIAN 
# This part reads "data" folder so the AI knows real SAT questions
def get_sat_context(user_query):
    context = ""
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return "No SAT Bank files found yet."

    # This looks at every .txt file in your /data folder
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                # If the student's question is in this file, we use it!
                if any(word.lower() in content.lower() for word in user_query.split()):
                    context += f"\n--- From {filename} ---\n{content[:1000]}\n"
    
    return context if context else "Use your general SAT knowledge to help."

# 4. UI & CAMERA
st.set_page_config(page_title="SAT AI Tutor", page_icon="🦉")
st.title("🦉 SAT Super Tutor")

# Camera Input
with st.expander("📸 Scan a Problem with Camera"):
    img_file = st.camera_input("Take a photo of your SAT question")
    if img_file:
        st.image(img_file)
        if st.button("Analyze Photo"):
            # Use Gemini to "Read" the image
            vision_model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = img_file.getvalue()
            response = vision_model.generate_content(["Describe this SAT math/english problem and explain the first step to solve it Socratically.", {"mime_type": "image/jpeg", "data": img_data}])
            st.write(response.text)

# 5. THE RULE BOOK (We move this to the top so the AI knows its job immediately)
# I have added "English Tutoring" instructions here for you!
system_prompt = """
ROLE:
You are an expert SAT Tutor for both Math and English. 

THE GOLDEN RULE:
Use the Socratic Method. Never give answers. Ask one small guiding question at a time.

ENGLISH TUTORING RULES:
1. If the student shares a reading passage, ask them: "What do you think the author's main point is?"
2. FEEDBACK: If a student provides an answer or a step, GIVE YOUR OPINION. 
   - If they are correct, say "Spot on!" and explain WHY they are right before asking a follow-up.
   - If they are incorrect, say "I see your thinking, but let's look closer at..." and ask a question about their specific error.
3. If they ask about grammar, ask: "Can you find the subject and the verb in this sentence?"
4. Always ask the student to provide 'evidence' from the text for their choice.

MATH TUTORING RULES:
1. Ask the student for the first step to solve the problem.
2. If they are stuck, give a hint about the math rule, not the number.
3. FEEDBACK: If a student provides an answer or a step, GIVE YOUR OPINION. 
   - If they are correct, say "Spot on!" and explain WHY they are right before asking a follow-up.
   - If they are incorrect, say "I see your thinking, but let's look closer at..." and ask a question about their specific error.

DISCLAIMER:
You must occasionally remind the student that while you are an advanced AI, you can make mistakes and they should verify critical steps.

TONE:
Encouraging, patient, and professional. 🦉✨
"""

# 6. HELPER FUNCTION (For the download button)
def convert_chat_to_text(messages):
    chat_log = "SAT TUTOR STUDY SESSION\n" + "="*25 + "\n\n"
    for msg in messages:
        if msg["role"] != "system":
            speaker = "Tutor" if msg["role"] == "assistant" else "Student"
            chat_log += f"{speaker}: {msg['content']}\n\n"
    return chat_log

# 7. INITIALIZE MEMORY (This MUST happen before the sidebar uses it)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": system_prompt}]

# 8. THE WEBSITE INTERFACE
st.set_page_config(page_title="SAT AI Tutor", page_icon="🦉")
st.title("🦉 Universal SAT Tutor")
st.caption("I help with Math, Reading, and Writing. I guide, you learn.")

# 9. SIDEBAR TOOLS
with st.sidebar:
    st.header("Study Tools")
    full_chat_text = convert_chat_to_text(st.session_state["messages"])
    st.download_button(
        label="📥 Download Study Log",
        data=full_chat_text,
        file_name="sat_study_session.txt",
        mime="text/plain"
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state["messages"] = [{"role": "system", "content": system_prompt}]
        st.rerun()
    with st.sidebar:
        st.header("📸 Image Tutor")
    uploaded_file = st.file_uploader("Upload or Paste an SAT question", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Show a preview so the student knows it worked
        st.image(uploaded_file, caption="Uploaded Problem", use_container_width=True)
        
        if st.button("Analyze Photo"):
            try:
                # 1. Initialize the 'Eyes'
                vision_model = genai.GenerativeModel('gemini-2.5-flash')
                img_data = uploaded_file.getvalue()
                
                with st.spinner("The Owl is reading your photo..."):
                    # 2. Get the transcript of the image
                    response = vision_model.generate_content([
                        "Extract all text, equations, and multiple-choice options from this SAT problem.",
                        {"mime_type": "image/jpeg", "data": img_data}
                    ])
                    
                    # 3. FEED THE BRAIN: Add the image text to the hidden chat history
                    # This is how the AI knows what 'C' refers to later!
                    image_text = response.text
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": f"[SYSTEM NOTE: Student uploaded a photo. Content: {image_text}]"
                    })
                    
                    # 4. Post a friendly message to the UI
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "📸 I've scanned your photo! Based on the problem I see, what's your initial thought on how to approach this?"
                    })
                    st.rerun()

            except Exception as e:
                # This is the part your code was missing!
                st.error(f"The Librarian hit a snag: {e}")
                st.info("Try waiting 10 seconds or check if your Gemini API key is correct.")
                
                # Add Gemini's observation to the chat history
            st.session_state.messages.append({"role": "assistant", "content": f"📸 I've looked at your image: {response.text}"})
            st.rerun() # Refresh to show the message in the chat

# 10. SHOW PREVIOUS CHAT
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# 11. THE CONVERSATION LOOP
if user_input := st.chat_input("Paste a math problem or an English passage here..."):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get answer from the brain
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state["messages"],
            temperature=0.7,
        )
        response = stream.choices[0].message.content
        st.write(response)
    
    # Save tutor message

    st.session_state["messages"].append({"role": "assistant", "content": response})
