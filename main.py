import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in .env file!")
    exit()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

print("\n" + "="*50)
print("LangGraph Terminal Bot Ready!")
print("Type 'quit' or 'exit' to stop.")
print("="*50)

config = {"configurable": {"thread_id": "terminal_session_1"}}

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Shutting down... Goodbye!\n")
        break

    for chunk in app.stream({"messages": [("user", user_input)]}, config=config, stream_mode="values"):
        latest_message = chunk["messages"][-1]
        
        if latest_message.type == "ai":
            print(f"Bot: {latest_message.content}")