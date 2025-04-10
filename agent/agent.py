# agent.py
"""
Agentic AI Recruiting Assistant using LangGraph.
This module builds the recruiting workflow as a cyclic graph.
It dynamically extracts job roles from the recruiter's input.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agent.tools import write_outreach_email, generate_checklist, google_web_search, generate_offer_letter, edit_content
from dotenv import load_dotenv
import os
from llm_config import llm

# # Configure Gemini with your API key
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")

memory = MemorySaver()

# "recruiter_info" stores details like roles, budget, and timeline.
class RecruiterState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    recruiter_info: Dict[str, Any]
    clarification_questions: List[str]

    wants_tool_chat: bool
    done_with_tools: bool

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
        f"You are a Hiring Assistant tasked with generating comprehensive job descriptions.\n"
        f"Extract job roles mentioned in this hiring request: '{state['messages'][-1].content}'. "
        f"Then, for each role, generate important follow-up questions to help create a job description."
        f"To ensure accuracy, please begin by asking questions about these compulsory topics and make sure to adapt them to be dynamic to the role:\n"
        f"Only give a bullet list of questions and nothing else"
        f"- Essential skills required for this role.\n"
        f"- Qualifications candidates need to have.\n"
        f"- Years of experience are required.\n"
        f"- What is the role level? (e.g., junior, mid-level, senior)\n"
        f"- What is the location of the job?(remote, hybrid(address), on-site(address))"
        f"- What is the budget or compensation range for this position?\n"
        f"Once you have listed these mandatory questions, proceed to list additional, dynamic optional questions that are relevant to the specific role.\n"
        f"These MAY include inquiries about:\n"
        f"- Specific responsibilities, Required certifications, Company culture/values, etc.\n"
        f"LIMIT the questions to maximum 10"
        f"- Any other details that would help tailor the job description more precisely\n"
        f"Make sure all compulsory questions are listed before moving on to the optional ones, and adjust follow-up questions based on the context of the role.\n"
        f"Output ONLY the questions in a bullet list format."
    )
    response_text = llm.invoke([HumanMessage(content=prompt)]).content
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
        f"Include required skills, responsibilities, Job location, and qualifications in markdown format."
        f"You can fill in other information based on the {clarifications}, {role} and general assumptions. Do not ask in the description to fill details."
        f"Avoid creating sections of which the user did not provide details."
    )

    # LOG: Print what you're sending to Gemini
    # print("Prompt sent to LLM:\n", prompt)
    try:
        jd_text = llm.invoke([HumanMessage(content=prompt)]).content
        #print("LLM responded with:\n", jd_response.content)  # LOG: Show the JD
        jd_msg = AIMessage(content=jd_text)
        return {"messages": [jd_msg]}
    except Exception as e:
        #print(" LLM error during JD generation:", str(e))
        return {"messages": [AIMessage(content="Something went wrong generating the job description.")]}

def final_node(state: RecruiterState):
    """
    Compile all outputs into a final recruiting plan summary.
    """
    summary = "Final Recruiting Plan:\n\n"
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            summary += msg.content + "\n\n"
    return {"messages": [AIMessage(content=summary)]}

# agent.py  ──────────────────────────────────────────────
def route_after_jd(state: RecruiterState) -> str:
    """
    After the JD is generated, jump to the tool chat only if the user
    explicitly asked for it (wants_tool_chat == True).
    """
    return "tool_node" if state.get("wants_tool_chat", False) else "final"


def route_after_tool(state: RecruiterState) -> str:
    """
    Stay in the tool loop until the user says they're done.
    """
    return "final" if state.get("done_with_tools", False) else "tool_node"


tool_node = ToolNode(tools=[write_outreach_email, generate_checklist, google_web_search, generate_offer_letter, edit_content])


# -------------------------------
# Build the Main Graph (Starts at 'initial')
# -------------------------------
builder1 = StateGraph(RecruiterState)
builder1.add_node("initial", initial_node)

builder1.add_edge(START, "initial")
builder1.set_finish_point("initial")  # 👈 This is key
graph = builder1.compile(checkpointer=memory)


# -------------------------------
# Build the Clarification Graph (Starts at 'clarification')
# -------------------------------
builder2 = StateGraph(RecruiterState)
# builder2.add_node("initial", initial_node)
builder2.add_node("clarification", clarification_node)
builder2.add_node("jd", jd_generation_node)
builder2.add_node("tool_node", tool_node)
builder2.add_node("final", final_node)

builder2.add_edge(START, "clarification")  # Start here instead
builder2.add_edge("clarification", "jd")
builder2.add_conditional_edges(
    "jd",
    route_after_jd,
    path_map={"tool_node": "tool_node", "final": "final"},
)
builder2.add_conditional_edges(
    "tool_node",
    route_after_tool,
    path_map={"tool_node": "tool_node", "final": "final"},
)
builder2.set_finish_point("final")
clarification_graph = builder2.compile(checkpointer=memory)


# Helper to start from beginning
def run_role_to_questions(user_input: str, thread_id: str):
    state = {
        "messages": [HumanMessage(content=user_input)],
        "recruiter_info": {},
        "clarification_questions": []
    }
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config=config)

def run_from_clarification(clarification_response: str, role: str, thread_id: str):
    state = {
        "messages": [HumanMessage(content=clarification_response)],
        "recruiter_info": {
            "clarifications": clarification_response,
            "role": role
        },
        "clarification_questions": []
    }
    config = {"configurable": {"thread_id": thread_id}}
    return clarification_graph.invoke(state, config=config)



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
