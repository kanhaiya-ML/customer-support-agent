from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from support_workflow import workflow
from support_workflow import escalate_agent
import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","null"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)

class Request(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    query: str
    solved: str

@app.post("/ask")
def ask(request: Request):
    output = workflow.invoke({
    "query": request.question,
    "result": "",
    "solved": "no",
    "next_agent": "",
    "final_answer": ""
})
    return {"answer": output['result']}


@app.post("/feedback")
def feedback(request: FeedbackRequest):

    if request.solved == "no":
        # result = escalate_agent({})
        return {"answer": "Connecting you to a human agent. Please wait..."}
    return {"answer": "Thank you for your feedback! Glad we could help."}


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

@app.post("/human-agent")
def human_agent(request: Request):
    prompt = f"""You are a human customer support agent named Alex. 
    Be empathetic, professional, and helpful.
    Never reveal you are an AI.
    Only answer Related to actual customer support questions don't engage with unrelated questions.
    if you struggle to find answer just write i simple apology and say to we are looking into your problem and reach out shortly.
    
    Customer issue: {request.question}"""

    response = llm.invoke(prompt)
    return {"answer": response.content}
