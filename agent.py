# agent.py
"""
Agentic AI Recruiting Assistant using LangGraph.
This module builds the recruiting workflow as a cyclic graph.
It dynamically extracts job roles from the recruiter’s input.
"""


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.prebuilt import ToolNode
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from tools import write_outreach_email, generate_checklist, google_web_search, generate_offer_letter, edit_content
from config import GEMINI_KEY

from llm_config import llm

# # Configure Gemini with your API key
# genai.configure(api_key=GEMINI_KEY)  # <-- Replace with your actual API key

# # Initialize Gemini model
# model = genai.GenerativeModel("gemini-1.5-flash")

# # Wrapper to mimic LangChain's LLM interface
# class GeminiWrapper:
#     def invoke(self, messages):
#         prompt = "\n".join([msg.content for msg in messages])
#         response = model.generate_content(prompt)
#         return AIMessage(content=response.text)


# Add memory setup
memory = ConversationBufferMemory(return_messages=True)

# Wrap your Gemini LLM in a conversation chain
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory
)


# "recruiter_info" stores details like roles, budget, and timeline.
class RecruiterState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    recruiter_info: Dict[str, Any]
    clarification_questions: List[str]

# -------------------------------
# Node Functions
# -------------------------------

def initial_node(state: RecruiterState):
    """
    Extract job roles from the recruiter's query dynamically using the LLM,
    then ask clarifying questions. If clarifications are already present, skip this step.
    """
    # Skip role/question extraction if we're already resuming from clarification
    if state["recruiter_info"].get("clarifications"):
        return {"messages": []}

    # Otherwise, extract roles and generate questions
    prompt = (
        f"Extract job roles mentioned in this hiring request: '{state['messages'][-1].content}'. "
        f"Then, for each role, generate important follow-up questions to help create a job description."
        f"Output ONLY the questions in a bullet list format."
        # f"Only maximum 3 questions no more that three most important questions, that is a rule."
    )
    response_text = conversation_chain.predict(input=prompt)
    questions = [line.strip("-• ").strip() for line in response_text.strip().split("\n") if line.strip()]
    state["clarification_questions"] = questions
    return {
        "messages": [AIMessage(content="To create the best job descriptions, please answer:\n\n" + "\n".join(f"- {q}" for q in questions))],
        "clarification_questions": questions,
    }
  

def clarification_node(state: RecruiterState):
    """
    Process the clarifying response and update recruiter_info.
    """
    user_response = state["messages"][-1].content.strip()
    state["recruiter_info"]["clarifications"] = user_response
    ack = SystemMessage(content="Thanks for the details. Generating job description draft now...")
    return {"messages": [ack]}

def jd_generation_node(state: RecruiterState):
    """
    Generate a job description draft based on recruiter_info.
    """
    info = state["recruiter_info"]
    clarifications = info.get("clarifications", "")
    role = info.get("role", "")
    prompt = (
        f"Generate a detailed job description for the role of {role} "
        f"based on the following clarifications:\n\n"
        f"{clarifications}\n\n"
        f"Include responsibilities, Job location, qualifications, and benefits in markdown format."
    )
    # LOG: Print what you're sending to Gemini
    print("📤 Prompt sent to LLM:\n", prompt)
    try:
        jd_text = conversation_chain.predict(input=prompt)
        #print("✅ LLM responded with:\n", jd_response.content)  # LOG: Show the JD
        jd_msg = AIMessage(content=jd_text)
        return {"messages": [jd_msg]}
    except Exception as e:
        #print("❌ LLM error during JD generation:", str(e))
        return {"messages": [AIMessage(content="Something went wrong generating the job description.")]}

# def checklist_node(state: RecruiterState):
#     """
#     Generate a hiring checklist.
#     """
#     roles = state["recruiter_info"].get("clarifications", "")
#     prompt = (
#         f"Based on this hiring plan: {roles}\n"
#         f"Create a startup hiring checklist: sourcing, screening, interviewing, onboarding."
#     )
#     checklist_text = conversation_chain.predict(input=prompt)
#     return {"messages": [AIMessage(content=checklist_text)]}

# def email_node(state: RecruiterState):
#     """
#     Draft an outreach email for potential candidates.
#     """
#     clar = state["recruiter_info"].get("clarifications", "")
#     prompt = f"Draft a friendly outreach email to candidates based on the following job planning info:\n{clar}"
#     email_text = conversation_chain.predict(input=prompt)
#     return {"messages": [AIMessage(content=email_text)]}

def final_node(state: RecruiterState):
    """
    Compile all outputs into a final recruiting plan summary.
    """
    summary = "Final Recruiting Plan:\n\n"
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            summary += msg.content + "\n\n"
    return {"messages": [AIMessage(content=summary)]}

# -------------------------------
# Conditional Functions
# -------------------------------

def branch_after_initial(state: RecruiterState) -> str:
    """
    If the initial node produced a clarifying question, branch to clarification.
    Otherwise, proceed to job description generation.
    """
    return "clarification"

def branch_after_jd(state: RecruiterState) -> str:
    """
    After JD generation, decide whether to generate a checklist or draft an email.
    A simple heuristic is used.
    """
    last_ai = state["messages"][-1].content.lower()
    if "responsibilities" in last_ai:
        return "checklist"
    return "email"


tool_node = ToolNode(tools=[write_outreach_email, generate_checklist, google_web_search, generate_offer_letter, edit_content])


# -------------------------------
# Build the Main Graph (Starts at 'initial')
# -------------------------------
builder1 = StateGraph(RecruiterState)
builder1.add_node("initial", initial_node)

builder1.add_edge(START, "initial")
builder1.set_finish_point("initial")  # 👈 This is key
graph = builder1.compile()


# -------------------------------
# Build the Clarification Graph (Starts at 'clarification')
# -------------------------------
builder2 = StateGraph(RecruiterState)
builder2.add_node("initial", initial_node)
builder2.add_node("clarification", clarification_node)
builder2.add_node("jd", jd_generation_node)
builder2.add_node("tool_chat", tool_node)
builder2.add_node("final", final_node)

builder2.add_edge(START, "clarification")  # <-- Start here instead
builder2.add_edge("clarification", "jd")
builder2.add_edge("final", "tool_chat")
builder2.add_edge("jd", "final")

builder2.set_finish_point("final")
clarification_graph = builder2.compile()


# Helper to start from beginning
def run_role_to_questions(user_input: str):
    state = {
        "messages": [HumanMessage(content=user_input)],
        "recruiter_info": {},
        "clarification_questions": []
    }
    return graph.invoke(state)

def run_from_clarification(clarification_response: str, role: str):
    state = {
        "messages": [HumanMessage(content=clarification_response)],
        "recruiter_info": {
            "clarifications": clarification_response,
            "role": role
        },
        "clarification_questions": []
    }
    return clarification_graph.invoke(state)



if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    # Simulate instantaneous workflow:
    # Recruiter enters a query dynamically.
    initial_input = {
        "messages": [HumanMessage(content="I need to hire a data scientist and a marketing manager.")],
        "recruiter_info": {}
    }
    # Run the initial node to extract roles.
    state_after_initial = graph.invoke(initial_input)
    
    # If clarifying questions were generated, simulate a clarifying response.
    if any(isinstance(m, HumanMessage) and ("budget" in m.content.lower() or "timeline" in m.content.lower())
           for m in state_after_initial["messages"]):
        clar_response = HumanMessage(content="Budget: $100k-$120k, Timeline: 3 months")
        state_after_clar = graph.invoke({
            "messages": state_after_initial["messages"] + [clar_response],
            "recruiter_info": state_after_initial.get("recruiter_info", {})
        })
    else:
        state_after_clar = state_after_initial

    # Continue execution until the final state.
    final_state = graph.invoke({
        "messages": state_after_clar["messages"],
        "recruiter_info": state_after_clar.get("recruiter_info", {})
    })
    
    # Print the final recruiting plan summary.
    # for msg in final_state["messages"]:
    #     print(msg.content)
