"""
AI Study Chatbot — Streamlit Web App
------------------------------------
ChatGPT-style web UI for the AI Study Mentor chatbot.
Reuses the same Gemini setup as main.py (model, system prompt, API pattern).
"""

# ============================================
# SECTION 1: IMPORTS
# ============================================

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================================
# SECTION 2: CONFIGURATION (shared with main.py)
# ============================================

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an expert AI Study Mentor.

Your job is to:
- teach students clearly
- explain concepts simply
- provide roadmap guidance
- help students stay motivated
- answer in beginner-friendly language

Rules:
- Keep answers practical and easy to understand
- Avoid overly technical jargon unless necessary
- Give structured and useful responses
- Encourage learning and curiosity
"""

APP_TITLE = "AI Study Chatbot"
APP_SUBTITLE = "Your personal AI Study Mentor — ask anything about learning, tech, and growth."


# ============================================
# SECTION 3: HELPER FUNCTIONS
# ============================================

def load_api_key() -> str:
    """Load GEMINI_API_KEY from .env securely."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    return api_key


@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    """Create one Gemini client for the app session (cached)."""
    return genai.Client(api_key=api_key)


def init_session_state() -> None:
    """Initialize chat history in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat_history() -> None:
    """Remove all messages from session state."""
    st.session_state.messages = []


def build_full_prompt(messages: list[dict]) -> str:
    """
    Build the prompt sent to Gemini.
    Same idea as main.py: system prompt + student content.
    Includes prior turns so follow-up questions stay in context.
    """
    conversation_lines = []
    for msg in messages:
        label = "Student" if msg["role"] == "user" else "Mentor"
        conversation_lines.append(f"{label}: {msg['content']}")

    conversation = "\n\n".join(conversation_lines)

    return f"""
{SYSTEM_PROMPT}

Conversation:
{conversation}
"""


def generate_ai_response(client: genai.Client, messages: list[dict]) -> str:
    """
    Call Gemini and return assistant text.
    Uses the same client.models.generate_content pattern as main.py.
    """
    full_prompt = build_full_prompt(messages)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
    )

    return response.text or "I could not generate a response. Please try again."


def apply_custom_styles() -> None:
    """Minimal ChatGPT-inspired layout and typography."""
    st.markdown(
        """
        <style>
        /* Center main chat column */
        .block-container {
            max-width: 48rem;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        /* Title area */
        .app-header h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .app-header p {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: 0;
        }

        /* Sidebar polish */
        section[data-testid="stSidebar"] {
            background-color: #f9fafb;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        /* Chat input bar */
        [data-testid="stChatInput"] {
            max-width: 48rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """App title and subtitle."""
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Sidebar: model info, clear chat, app details."""
    with st.sidebar:
        st.header("Settings")

        st.subheader("Model")
        st.info(f"**{MODEL_NAME}**\n\nGoogle Gemini via `google-genai` SDK")

        st.divider()

        if st.button("Clear chat", use_container_width=True, type="primary"):
            clear_chat_history()
            st.rerun()

        st.divider()

        st.subheader("About")
        st.markdown(
            """
            **AI Study Chatbot** helps you learn with clear,
            beginner-friendly explanations.

            - Built with **Streamlit**
            - Powered by **Google Gemini**
            - API key stored in **`.env`**

            Type your question below to start.
            """
        )


def render_chat_history() -> None:
    """Show all messages as user / assistant chat bubbles."""
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])


def handle_user_input(client: genai.Client) -> None:
    """Read chat input, call Gemini with spinner, append assistant reply."""
    user_prompt = st.chat_input("Ask your study question...")

    if not user_prompt:
        return

    if not user_prompt.strip():
        st.warning("Please enter a question.")
        return

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompt.strip()})

    with st.chat_message("user"):
        st.markdown(user_prompt.strip())

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = generate_ai_response(client, st.session_state.messages)
            except Exception as error:
                reply = (
                    "Something went wrong while generating a response. "
                    "Please check your API key and connection, then try again."
                )
                st.error(f"Error details: {error}")

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


# ============================================
# SECTION 4: STREAMLIT APP ENTRY POINT
# ============================================

def main() -> None:
    """Run the Streamlit web application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    init_session_state()

    try:
        api_key = load_api_key()
        client = get_gemini_client(api_key)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Failed to initialize Gemini client: {error}")
        st.stop()

    render_sidebar()
    render_header()
    render_chat_history()
    handle_user_input(client)


if __name__ == "__main__":
    main()
