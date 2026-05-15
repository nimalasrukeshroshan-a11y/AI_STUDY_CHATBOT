"""
AI Study Chatbot using Gemini API
---------------------------------
A beginner-friendly AI chatbot built using:
- Python
- Google Gemini API
- Prompt Engineering
- Environment Variables

Features:
- Secure API key handling
- AI Study Mentor personality
- Continuous conversation loop
- Beginner-friendly structure
"""

# ============================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================

import os

from dotenv import load_dotenv
from google import genai


# ============================================
# SECTION 2: LOAD ENVIRONMENT VARIABLES
# ============================================

# Load variables from .env file
load_dotenv()


# ============================================
# SECTION 3: GET GEMINI API KEY
# ============================================

# Read API key securely from .env
api_key = os.getenv("GEMINI_API_KEY")

# Stop program if key is missing
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================
# SECTION 4: CREATE GEMINI CLIENT
# ============================================

# Create Gemini client connection
client = genai.Client(api_key=api_key)


# ============================================
# SECTION 5: SELECT MODEL
# ============================================

# Latest stable Gemini model
MODEL_NAME = "gemini-2.5-flash"


# ============================================
# SECTION 6: SYSTEM PROMPT
# ============================================

# This controls chatbot personality and behavior
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


# ============================================
# SECTION 7: WELCOME MESSAGE
# ============================================

print("\n===================================")
print("      AI STUDY CHATBOT READY")
print("===================================")
print(f"Using Model: {MODEL_NAME}")
print("Type 'exit' or 'quit' to stop.\n")


# ============================================
# SECTION 8: CHATBOT LOOP
# ============================================

while True:

    # Take user input
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() in ["exit", "quit"]: #Exit #EXIT #QUIT #Quit
        print("\nChatbot: Goodbye! Keep learning AI 🚀")
        break

    # Handle empty input
    if not user_input:
        print("Chatbot: Please enter a question.\n")
        continue

    try:

        # Combine system prompt + user question
        full_prompt = f"""
        {SYSTEM_PROMPT}

        Student Question:
        {user_input}
        """

        # Generate AI response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )

        # Print chatbot response
        print(f"\nChatbot: {response.text}\n")

    except Exception as error:

        # Show friendly error message
        print("\nChatbot: Something went wrong.")
        print(f"Error Details: {error}\n")