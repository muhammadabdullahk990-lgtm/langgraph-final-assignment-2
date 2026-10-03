import os
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: Literal["RAG_Agent", "GitHub_MCP_Agent", "Workspace_MCP_Agent", "FINISH"]

# ==========================================
# 2. Sub-Agent 1: Hosted RAG Agent (PDF Vector Store)
# ==========================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "pdf-knowledge-base"),
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def search_pdf_knowledge_base(query: str) -> str:
    """Useful to search through uploaded and indexed PDF documentation in vector store."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant PDF passages found."
    return "\n\n---\n\n".join([f"Source: {d.metadata.get('source', 'PDF')}\nContent: {d.page_content}" for d in docs])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag_sub_agent = create_react_agent(
    model=llm,
    tools=[search_pdf_knowledge_base],
    state_modifier="You are a RAG retrieval expert. Search the hosted PDF vector database to answer queries."
)

# ==========================================
# 3. Sub-Agent 2: GitHub MCP Agent
# ==========================================
@tool
def github_search_repositories(query: str) -> str:
    """Search for GitHub repositories using MCP integration."""
    return f"[GitHub MCP] Found repositories matching query: '{query}'."

@tool
def github_read_file(owner: str, repo: str, path: str, branch: str = "main") -> str:
    """Fetch content of a file from a specified GitHub repository via MCP."""
    return f"[GitHub MCP] Content of {owner}/{repo}/{path} on branch {branch}."

@tool
def github_get_issues(owner: str, repo: str, state: str = "open") -> str:
    """Query issues and pull requests from a GitHub repository via MCP."""
    return f"[GitHub MCP] Active {state} issues and PRs for {owner}/{repo}."

github_sub_agent = create_react_agent(
    model=llm,
    tools=[github_search_repositories, github_read_file, github_get_issues],
    state_modifier="You are a GitHub MCP Specialist. Answer repository and code queries."
)

# ==========================================
# 4. Sub-Agent 3: Workspace MCP Agent (Calendar & Gmail)
# ==========================================
@tool
def calendar_list_events(time_min: str, time_max: str) -> str:
    """Read Google Calendar events and meetings within a specific time range."""
    return f"[Google Calendar MCP] Meetings from {time_min} to {time_max} returned."

@tool
def calendar_create_event(title: str, start_time: str, end_time: str, attendees: list[str]) -> str:
    """Create a Google Calendar meeting with title, timestamps, and attendees."""
    return f"[Google Calendar MCP] Meeting '{title}' scheduled for {start_time} with {attendees}."

@tool
def gmail_create_draft(recipient: str, subject: str, body: str) -> str:
    """Compose and save a draft email in Gmail."""
    return f"[Gmail MCP] Draft created for {recipient} with subject '{subject}'."

@tool
def gmail_send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email immediately through Gmail."""
    return f"[Gmail MCP] Email successfully sent to {recipient} with subject '{subject}'."

@tool
def schedule_meeting_and_email(attendee_email: str, title: str, start_time: str, end_time: str) -> str:
    """Automated workflow: schedules a meeting and immediately dispatches email invitation."""
    return f"[Workspace MCP] Meeting '{title}' confirmed at {start_time}. Invitation sent to {attendee_email}."

workspace_sub_agent = create_react_agent(
    model=llm,
    tools=[calendar_list_events, calendar_create_event, gmail_create_draft, gmail_send_email, schedule_meeting_and_email],
    state_modifier="You are a Google Workspace Assistant managing Calendar and Gmail."
)

# ==========================================
# 5. LangGraph Supervisor Orchestrator
# ==========================================
class Router(BaseModel):
    next: Literal["RAG_Agent", "GitHub_MCP_Agent", "Workspace_MCP_Agent", "FINISH"]

supervisor_prompt = (
    "You are a supervisor managing 3 sub-agents:\n"
    "1. RAG_Agent: For hosted PDF documentation questions.\n"
    "2. GitHub_MCP_Agent: For GitHub queries, code, repos, and issues.\n"
    "3. Workspace_MCP_Agent: For Google Calendar (read/schedule) and Gmail (draft/send).\n"
    "Select the appropriate agent to respond or FINISH when done."
)

def supervisor_node(state: AgentState):
    messages = [{"role": "system", "content": supervisor_prompt}] + list(state["messages"])
    response = llm.with_structured_output(Router).invoke(messages)
    return {"next": response.next}

def call_rag(state: AgentState):
    res = rag_sub_agent.invoke({"messages": state["messages"]})
    return {"messages": [res["messages"][-1]]}

def call_github(state: AgentState):
    res = github_sub_agent.invoke({"messages": state["messages"]})
    return {"messages": [res["messages"][-1]]}

def call_workspace(state: AgentState):
    res = workspace_sub_agent.invoke({"messages": state["messages"]})
    return {"messages": [res["messages"][-1]]}

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("RAG_Agent", call_rag)
workflow.add_node("GitHub_MCP_Agent", call_github)
workflow.add_node("Workspace_MCP_Agent", call_workspace)

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "RAG_Agent": "RAG_Agent",
        "GitHub_MCP_Agent": "GitHub_MCP_Agent",
        "Workspace_MCP_Agent": "Workspace_MCP_Agent",
        "FINISH": END
    }
)

workflow.add_edge("RAG_Agent", "supervisor")
workflow.add_edge("GitHub_MCP_Agent", "supervisor")
workflow.add_edge("Workspace_MCP_Agent", "supervisor")
workflow.add_edge(START, "supervisor")

app = workflow.compile()
