from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from langgraph.checkpoint.memory import MemorySaver
from model import predict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from rag import hybrid_search
import os
from dotenv import load_dotenv
load_dotenv()


checkpointer = MemorySaver()

class SupportState(TypedDict):
    query: str
    result: str
    solved: Literal["yes","no"]
    next_agent: str
    final_answer: str
    messages: Annotated[list[BaseMessage],add_messages]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def choose_agent(state: SupportState):
    prompt = f"""Read user query and decide which agent to use.
    'rag' - for questions about company policies, refunds policy, return policy, offers.
    'web' - for company current status in market what is growth and insights of the company. 
    'generic_agent' - when user ask about there order, status, account related issue or payment related issue then use this.
    'escalate' - ONLY when user explicitly says 'talk to human' or 'speak to agent'
    Return only one word: rag, web, escalate or generic.
    user query: {state['query']}"""
    response = llm.invoke(prompt)
    return {"next_agent":response.content.strip().lower()}

def rag_agent(state: SupportState):
    docs = hybrid_search.invoke(state["query"])

    context = "\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = f"""
    You are a customer support assistant.
    Answer ONLY from the provided context.
    Context:
    {context}
    Question:
    {state['query']}
    If the answer is not present in the context,
    say "I couldn't find that information."
    """
    response = llm.invoke(prompt)
    return {
        "result": response.content
    }


def web_agent(state: SupportState):
    with DDGS() as ddgs:
        results = list(ddgs.text(state['query'], max_results=3))
    output = "\n".join([r['body'] for r in results])
    return {"result":output}

def escalate_agent(state: SupportState):
      return {"result": "My name is Kanhaiya and I'm here to solve your query. Please describe your issue."}

def generic_agent(state: SupportState):
    # response = predict(state['query'])
    # return {"result":response}
    prompt = f"""You are a customer support agent reply on user query only related to oredrs and any type of actuall issues.
    i am giving you the conversation history between you and customer keep replies short and simple.
    conversation history: {state['messages']}\ncurrent query: {state['query']}.
    return a simple in context replies and after getting information
    just reply we are looking into it your issue will be solved in some hours"""

    response = llm.invoke(prompt)
    return {"result": response.content}


def agent_router(state: SupportState):
    return state["next_agent"]


graph = StateGraph(SupportState)
graph.add_node("choose_agent",choose_agent)
graph.add_node("rag_agent",rag_agent)
graph.add_node("web_agent",web_agent)
graph.add_node("escalate_agent",escalate_agent)
graph.add_node("generic_agent",generic_agent)
# graph.add_node("responder",responder)

graph.add_edge(START,"choose_agent")
graph.add_conditional_edges("choose_agent",agent_router,{"web":"web_agent","rag":"rag_agent","escalate":"escalate_agent","generic":"generic_agent"})
# graph.add_conditional_edges("responder",router,{"yes":END,"no":"escalate_agent"})
graph.add_edge("rag_agent", END)
graph.add_edge("web_agent", END)
graph.add_edge("escalate_agent", END)

workflow = graph.compile(checkpointer=checkpointer)
