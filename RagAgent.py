from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pdf_path = "Market_Analysis.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError("PDF with given filename doesn't exist")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Pdf loaded with {len(pages)} pages")
except:
    print(f"Error loading pdf.")
    raise

text_spiltter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_spilt = text_spiltter.split_documents(pages)

persist_directory = "/Users/shivamkhokhani/Desktop/Projects/RAGAgent"
collection = 'market_analysis'

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_spilt,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection,
    )
    print(f"Chroma VectorDB created successfully!")
except:
    print(f"Error creating vectorDB")
    raise

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs = {"k":5}
)

@tool 
def retriever_tool(query: str):
    """ This tool retrieves the relevant information about the query from the Market Analysis document."""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information was found in the document related to the query"
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n {doc}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState):
    """ This function check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Market Analysis based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tool_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm_agent(state: AgentState) -> AgentState:
    """ This function calls the llm with current state"""

    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return { "messages" : [message]}


def retriever_agent(state: AgentState) -> AgentState:
    """ This function calls the appropriate tools as per the request from LLM"""

    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Executing tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tool_dict:
            print(f"No tool: {t['name']} exists.")
            result = "Incorrect tool. Please retry and select a tool from the available tools"
        else:
            result = tool_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Length of result: {len(str(result))}\n")
        
        results.append(ToolMessage(tool_call_id=t['id'], tool_name=t['name'], content=str(result)))
    
    print("Tool execution complete.\n")
    return {"messages" : results}


graph = StateGraph(AgentState)

graph.add_node("llm", call_llm_agent)
graph.add_node("retriever", retriever_agent)

graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:'retriever',
        False: END 
    }
)

graph.add_edge("retriever", "llm")

rag_agent = graph.compile()


def run_agent():
    print("====RAG AGENT====")

    while True:
        user_input = input("\nHow can I help you?")
        if user_input.lower() in ['exit', 'quit']:
            break 
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("======ANSWER======")
        print(result['messages'][-1].content)

run_agent()




