import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun #DuckDuckGoSearchRun is supports to search anything frominternet
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.tools.retriever import create_retriever_tool

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool

# """ 
# About StreamlitCallbackHandler 

# It shows what the agent is doing in real-time — like:

# Which tool it is calling

# What input it sent to the tool

# What result it got

# What it’s thinking next

# 🧠 Without It:
# You only see the final answer.

# 👀 With StreamlitCallbackHandler:
# You can watch each step — great for debugging or transparency.


# """
import os
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

st.secrets["COHERE_API_KEY"]
#arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#wiki tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

#internetSearch tool
search = DuckDuckGoSearchRun(name="Search")

st.title("Langchain - Chat with search")



# sidebar for setting
# this code manasilvaahn refer 
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your Groq API keys:",type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192")

    system_prompt = (
        
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question."
            "{context}"
    )
    QA_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            ("human","{input}"),
        ]
    )

    if "vectors" not in st.session_state:
        st.session_state.embeddings=CohereEmbeddings(model='embed-english-light-v3.0',cohere_api_key=cohere_api_key,user_agent='langchain')
        st.session_state.loader=PyPDFLoader("attention.pdf") # data ingestion
        st.session_state.docs=st.session_state.loader.load() # document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings)
    
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm,QA_prompt)
    chain = create_retrieval_chain(retriever,document_chain)
    retriever_tool = Tool(
    name="Pdf-content",
    func=lambda q: chain.invoke({"input":q}),
    description = "Use this tool to answer questions specifically from the content of the uploaded PDF file. This is helpful when the user asks questions like 'Explain the attention mechanism in the document' or 'Summarize section 3 of the PDF'.",
) 

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role":"assisstant",
         "content":"Hi,I'm a chatbot who can search the web. how can i help you"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    
    tools = [search,arxiv_tool,wiki_tool,retriever_tool]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True) #AgnetType vereyum ind purpose nn refer vedio or gpt ,custom  ayitt prompt use [chatprompt okke] cheyynindengil Agenttype kodukanda 

    with st.chat_message("assistant"): # code manasilavuniilengil refer gpt not vedio, nalla reethiyil prompt cheythal in sha ALLAH manasilavum 
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
