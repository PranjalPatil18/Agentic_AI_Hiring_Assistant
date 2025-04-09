from llm_config import llm
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import google.generativeai as genai
from google.generativeai import types
from config import GEMINI_KEY

# tools.py
from langchain.tools import tool

@tool
def write_outreach_email(purpose: str = "interview", candidate: str = "the candidate") -> str:
    """Use LLM to write a friendly outreach email for a specified purpose (e.g., interview, follow-up)."""
    prompt = (
        f"Write a friendly outreach email to {candidate} for the purpose of {purpose}. "
        "Keep it professional, clear, and concise. Use a warm tone."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

@tool
def generate_checklist(context: str) -> str:
    """Generate a hiring checklist based on the hiring clarifications (role, responsibilities, etc)."""
    prompt = (
        f"Based on this hiring plan: {context}\n"
        f"Create a startup hiring checklist. Break it down into stages like sourcing, screening, interviewing, onboarding."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

@tool
def google_web_search(query: str) -> str:
    """Perform a real-time web search using Gemini's Grounding with Google Search."""
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    return response.candidates[0].content.parts[0].text

@tool
def generate_offer_letter(candidate_name: str, salary: str) -> str:
    """Generate a professional offer letter given a candidate's name and salary."""
    prompt = (
        f"Create a formal offer letter for {candidate_name} for the offered position. "
        f"The annual salary is {salary}. Include details like joining date (to be discussed), "
        f"benefits, company culture, and a welcoming tone. Format it professionally."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

@tool
def edit_content(existing: str, instruction: str) -> str:
    """Edit a piece of content (JD, email, checklist, etc.) based on user instructions."""
    prompt = (
        f"You are a professional assistant. Your job is to edit content according to recruiter needs.\n\n"
        f"Here is the existing content:\n{existing}\n\n"
        f"Here is the instruction to modify it:\n{instruction}\n\n"
        f"Update the content accordingly. Return only the updated version."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

