import streamlit as st
import requests
import logging
import os
import uuid

# Configure logging for Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BACKEND_URL = os.getenv("POLICYPILOT_BACKEND_URL", "http://localhost:8000/api").rstrip("/")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PolicyPilot AI",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- MAIN CHAT INTERFACE ---
# Create two columns
col1, col2 = st.columns([0.5, 4])  # Adjust the width ratio as needed
with col1:
    st.image("PolicyPilot_Logo.png", width=70)  # Adjust width as needed
with col2:
    st.title("PolicyPilot")
st.caption("Your AI guide to U.S. Health Insurance")

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes all necessary keys in the session state."""
    if "current_phase" not in st.session_state:
        st.session_state.current_phase = "initial_zip"
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    # (CRITICAL FIX) Initialize the profile completion flag
    if "is_profile_complete" not in st.session_state:
        st.session_state.is_profile_complete = False
    if "sources" not in st.session_state:
        st.session_state.sources = []

initialize_session_state()

# --- Helper Functions for API Calls ---
@st.cache_data
def get_geodata_from_backend(_zip_code: str):
    """Calls the FastAPI /geodata endpoint. Cached to prevent re-calls."""
    try:
        response = requests.get(f"{BACKEND_URL}/geodata/{_zip_code}", timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching location data: {e}")
        return None

def send_chat_message_to_backend(user_message: str):
    """Calls the single, unified FastAPI /chat endpoint."""
    payload = {
        "thread_id": st.session_state.thread_id,
        "user_profile": st.session_state.user_profile,
        "message": user_message,
        "conversation_history": st.session_state.chat_history,
        # (CRITICAL FIX) Pass the current profile completion status to the backend
        "is_profile_complete": st.session_state.is_profile_complete
    }
    logger.info("Sending PolicyPilot chat request for thread_id=%s", st.session_state.thread_id)
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with AI backend: {e}")
        return None

# --- UI Rendering Functions ---

def display_zip_form():
    st.header("1. Let's Start with Your Location")
    zip_code_input = st.text_input("Enter your 5-digit ZIP code:", max_chars=5)
    if st.button("Confirm ZIP Code"):
        if zip_code_input and len(zip_code_input) == 5 and zip_code_input.isdigit():
            with st.spinner("Verifying ZIP code..."):
                geo_data = get_geodata_from_backend(zip_code_input)
            if geo_data:
                st.session_state.user_profile.update(geo_data)
                st.session_state.current_phase = "basic_profile"
                st.rerun()
        else:
            st.error("Please enter a valid 5-digit ZIP code.")

def display_basic_profile_form():
    st.header("2. Tell Us More About You")
    with st.form("basic_profile_form"):
        age = st.number_input("Age", min_value=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        household_size = st.number_input("Household Size", min_value=1)
        income = st.number_input("Annual Household Income ($)", min_value=0, step=1000)
        employment_status = st.selectbox("Employment Status", ["Employed with employer coverage", "Employed without coverage", "Unemployed", "Retired", "Student", "Self-employed"])
        citizenship = st.selectbox("Citizenship Status", ["US Citizen", "Lawful Permanent Resident", "Other legal resident", "Non-resident"])

        submitted = st.form_submit_button("Start My Personalized Session")
        if submitted:
            st.session_state.user_profile.update({
                "age": age, "gender": gender, "household_size": household_size,
                "income": income, "employment_status": employment_status,
                "citizenship": citizenship,
                "medical_history": None, "medications": None, "special_cases": None
            })
            st.session_state.current_phase = "chat"
            # The chat interface will handle the START_PROFILE_BUILDING trigger
            st.rerun()

def display_chat_interface():
    st.header("3. Let's Chat!")

    # If this is the first time entering the chat phase, get the first question.
    if not st.session_state.chat_history and not st.session_state.is_profile_complete:
        with st.spinner("Starting your personalized profile conversation..."):
            response_data = send_chat_message_to_backend("START_PROFILE_BUILDING")
            if response_data:
                st.session_state.user_profile = response_data["updated_profile"]
                st.session_state.chat_history = response_data["updated_history"]
                st.session_state.is_profile_complete = response_data["is_profile_complete"]
                st.session_state.sources = response_data.get("sources", [])
                st.rerun()

    # Display chat history from session state
    for message in st.session_state.chat_history:
        # Handle potential errors if message format is incorrect
        if ":" in message:
            role, content = message.split(":", 1)
            with st.chat_message(role.lower()):
                st.markdown(content.strip())
        else:
            st.chat_message("system").write(message) # Fallback for unexpected format

    # User input
    if prompt := st.chat_input("Your message..."):
        # Display user message immediately
        st.chat_message("user").markdown(prompt)

        # Send to backend and get response
        with st.spinner("PolicyPilot AI is thinking..."):
            response_data = send_chat_message_to_backend(prompt)

        if response_data:
            # The backend is the source of truth for all state. Overwrite local state.
            st.session_state.user_profile = response_data["updated_profile"]
            st.session_state.chat_history = response_data["updated_history"]
            st.session_state.is_profile_complete = response_data["is_profile_complete"]
            st.session_state.sources = response_data.get("sources", [])
            st.rerun()

    if st.session_state.sources:
        with st.expander("Government sources used in the latest answer"):
            for source in st.session_state.sources:
                st.markdown(f"- [{source['name']}]({source['url']})")

# --- Main Application Flow Control ---
if st.session_state.current_phase == "initial_zip":
    display_zip_form()
elif st.session_state.current_phase == "basic_profile":
    display_basic_profile_form()
else: # 'chat' phase
    display_chat_interface()

st.caption(
    "PolicyPilot provides informational guidance, not an official eligibility or coverage "
    "determination. Confirm decisions with the relevant government program or insurer."
)

with st.sidebar:
    st.header("PolicyPilot")
    st.write("Your profile stays within this local demo session.")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()
