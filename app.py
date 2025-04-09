# app.py
import streamlit as st
import json
from llm_config import llm
from agent import graph, run_role_to_questions, run_from_clarification
from analytics import AnalyticsTracker
from memory import SessionMemory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools import write_outreach_email, generate_checklist, google_web_search, generate_offer_letter, edit_content
from langgraph.prebuilt import ToolNode
from agent import memory  # your ConversationBufferMemory

# -------------------------------
# Session Initialization
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "analytics_logs" not in st.session_state:
    st.session_state.analytics_logs = []
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "recruiter_info" not in st.session_state:
    st.session_state.recruiter_info = {}
if "clarification_questions" not in st.session_state:
    st.session_state.clarification_questions = []
if "clarification_answers" not in st.session_state:
    st.session_state.clarification_answers = {}
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "latest_tool_output" not in st.session_state:
    st.session_state.latest_tool_output = ""
# NEW: store every generated artifact so we can edit it later
if "generated" not in st.session_state:
    st.session_state.generated = {
        "jd": "",          # Job description
        "email": "",       # Outreach email
        "checklist": "",   # Hiring checklist
        "offer_letter": "" # Offer letter
    }

# -------------------------------
# Tool Setup
# -------------------------------

tool_enabled_llm = llm.bind_tools([
    write_outreach_email,
    generate_checklist,
    google_web_search,
    generate_offer_letter,
    edit_content
])

tool_node = ToolNode(tools=[
    write_outreach_email,
    generate_checklist,
    google_web_search,
    generate_offer_letter,
    edit_content
])

# -------------------------------
# Header
# -------------------------------

st.title("Agentic Hiring Assistant")
analytics = AnalyticsTracker()
st.sidebar.header("Usage Analytics")
st.sidebar.text(json.dumps(analytics.get_logs(), indent=2))

st.sidebar.header("Graph Visualization")
try:
    from flow import get_mermaid_diagram
    diagram = get_mermaid_diagram()
    st.sidebar.text(diagram)
except Exception as e:
    st.sidebar.text(f"Graph visualization error: {e}")

# -------------------------------
# Step 1: Enter Role
# -------------------------------

st.subheader("Step 1: Enter the Role You’re Hiring For")
with st.form("role_form"):
    role_input = st.text_input("e.g., I need to hire a founding engineer")
    submit_role = st.form_submit_button("Submit")

if submit_role and role_input:
    result_state = run_role_to_questions(role_input)
    st.session_state.conversation.extend([msg.content for msg in result_state["messages"]])
    st.session_state.recruiter_info = {"role": role_input}
    st.session_state.clarification_questions = result_state.get("clarification_questions", [])
    st.session_state.clarification_answers = {}
    st.session_state.jd_text = ""
    # reset generated store except role specific JD which will be generated later
    st.session_state.generated = {k: "" for k in st.session_state.generated}

# -------------------------------
# Step 2: Clarification
# -------------------------------

if st.session_state.clarification_questions:
    st.subheader("Step 2: Answer Clarification Questions")
    clarification_inputs = {}

    with st.form("clarification_form"):
        for q in st.session_state.clarification_questions:
            clarification_inputs[q] = st.text_input(label=q, key=f"input_{q}")
        generate_jd = st.form_submit_button("Generate JD")

    if generate_jd:
        st.session_state.clarification_answers = clarification_inputs
        user_responses = "\n".join(f"{q}: {clarification_inputs[q]}" for q in st.session_state.clarification_questions)

        try:
            result_state = run_from_clarification(user_responses, st.session_state.recruiter_info.get("role", ""))
            ai_messages = [msg.content for msg in result_state["messages"] if isinstance(msg, AIMessage)]

            if len(ai_messages) >= 2 and "Final Recruiting Plan" in ai_messages[-1]:
                ai_messages = ai_messages[:-1]

            jd_text = next((msg for msg in ai_messages if "Job Description" in msg or "##" in msg), ai_messages[0])
            st.session_state.jd_text = jd_text
            st.session_state.generated["jd"] = jd_text  # save JD for future edits

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# -------------------------------
# Step 3: Follow-Up Tool Chat
# -------------------------------

if st.session_state.jd_text:
    st.subheader("Generated Job Description")
    st.markdown(st.session_state.jd_text)

    st.subheader("Ask Follow-Up (e.g., 'generate email', 'edit checklist')")

    with st.form("followup_form"):
        user_followup = st.text_input("Your request:", key="followup_input")
        submit_followup = st.form_submit_button("Send")

    if submit_followup and user_followup:
        print("🧠 User follow-up:", user_followup)

        from langchain_core.messages import get_buffer_string

        # give the LLM the latest artifacts so it can choose what to edit
        system_prompt = f"""
            Here are the current artifacts you can reference or edit.

            [jd] Job Description:
            {st.session_state.generated['jd']}

            [email] Outreach Email:
            {st.session_state.generated['email']}

            [checklist] Hiring Checklist:
            {st.session_state.generated['checklist']}

            [offer_letter] Offer Letter:
            {st.session_state.generated['offer_letter']}

            • When the user wants to *change* one of these, call the tool **edit_content** with:
                - existing: the exact text of the item being edited
                - instruction: the user’s request
            • Otherwise, call the appropriate generation tool.
            """

        ai_msg = tool_enabled_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_followup)
        ])

        print("🤖 AIMessage content:", ai_msg.content)
        print("🛠️ Tool Calls:", getattr(ai_msg, "tool_calls", None))
        tool_calls = getattr(ai_msg, "tool_calls", None)
        tool_name = None
        if tool_calls:
            tool_name = tool_calls[0]['name']  # assumes one tool call

        input_state = {
            "messages": [ai_msg],
            "recruiter_info": {"clarifications": st.session_state.jd_text},
            "clarification_questions": []
        }
        # Attach chat history only if editing content
        if tool_name == "edit_content":
            input_state["history"] = memory.chat_memory.messages

        result = tool_node.invoke(input_state)
        print("📦 ToolNode result:", result)

        tool_response = next(
            (msg.content for msg in result["messages"] if isinstance(msg, ToolMessage)),
            ai_msg.content
        )

        # Update generated store based on which tool was invoked
        if tool_name == "write_outreach_email":
            st.session_state.generated["email"] = tool_response
        elif tool_name == "generate_checklist":
            st.session_state.generated["checklist"] = tool_response
        elif tool_name == "generate_offer_letter":
            st.session_state.generated["offer_letter"] = tool_response
        elif tool_name == "edit_content":
            lower_req = user_followup.lower()
            if "job description" in lower_req or "jd" in lower_req:
                st.session_state.generated["jd"] = tool_response
            elif "email" in lower_req:
                st.session_state.generated["email"] = tool_response
            elif "checklist" in lower_req:
                st.session_state.generated["checklist"] = tool_response
            elif "offer" in lower_req:
                st.session_state.generated["offer_letter"] = tool_response

        # Manually log tool result into memory so edit_content can access it later
        memory.chat_memory.add_user_message(tool_response)

        st.subheader("Tool Response")
        st.markdown(tool_response)
