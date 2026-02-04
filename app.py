import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
        color: #333333;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 10px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 2.8em;
        background-color: #2c3e50; 
        color: white;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border: 1px solid #f0f0f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1a1a1a;
    }
    
    /* Info Boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here.")
    if not api_key:
        st.warning("Please enter your API Key to proceed.")
        
    st.markdown("---")
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a TXT document", type=["txt"])
    
    if st.button("Clear History"):
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []
            st.rerun()

# --- Main Interface ---
st.title("🤖 Chat with your Document")
st.caption("Upload a document and ask questions about its content.")

# --- Caching Resource ---
@st.cache_resource(show_spinner=False)
def get_vectorstore(file_content, filename, api_key_val):
    """
    Process the document and create a vector store.
    Using Streamlit cache to avoid re-processing unless file changes.
    """
    if not api_key_val:
        return None
        
    try:
        # Create a temp file to load data (TextLoader requires a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        loader = TextLoader(temp_file_path)
        docs = loader.load()
        
        # Clean up temp file
        os.unlink(temp_file_path)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=api_key_val)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

# --- Application Logic ---
if uploaded_file and api_key:
    # Process the uploaded file
    file_content = uploaded_file.read()
    
    with st.spinner("Processing document..."):
        vector_store = get_vectorstore(file_content, uploaded_file.name, api_key)

    if vector_store:
        # Setup session state for history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # --- Retrieval Chain Setup ---
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key, streaming=True)
        retriever = vector_store.as_retriever()

        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # QA chain prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # --- Display Chat History ---
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

        # --- User Input ---
        user_query = st.chat_input("Ask a question about the document...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                # Stream the response
                response_container = st.empty()
                full_response = ""
                
                # Execute the chain
                result = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = result["answer"]
                response_container.write(answer)
                
                # Update history
                st.session_state.chat_history.extend(
                    [HumanMessage(content=user_query), AIMessage(content=answer)]
                )

else:
    if not uploaded_file:
        st.info("👋 Welcome! Please upload a text document in the sidebar to get started.")
    elif not api_key:
        st.info("👈 Please enter your OpenAI API key in the sidebar.")
