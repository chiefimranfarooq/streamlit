import streamlit as st
import openai
from pathlib import Path
import os
from typing import Optional
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "A professional document chat application powered by OpenAI"
    }
)

# ============================================================================
# CUSTOM THEME & STYLING
# ============================================================================
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
    }
    
    /* Custom styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .header-style {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    .subheader-style {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    .document-info {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize all session state variables."""
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    
    if "document_content" not in st.session_state:
        st.session_state.document_content = None
    
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0

initialize_session_state()

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================
@st.cache_data(ttl=3600)
def load_document_content(file_content: bytes, filename: str) -> str:
    """Load and cache document content."""
    try:
        content = file_content.decode('utf-8')
        return content
    except Exception as e:
        st.error(f"Error reading document: {str(e)}")
        return None

@st.cache_resource
def get_openai_client(api_key: str):
    """Create and cache OpenAI client."""
    return openai.OpenAI(api_key=api_key)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def build_system_prompt(document_content: str) -> str:
    """Build an optimized system prompt."""
    return f"""You are a professional document analysis assistant. 
Your role is to answer questions about the provided document accurately and concisely.

DOCUMENT CONTENT:
{document_content}

INSTRUCTIONS:
- Answer questions based ONLY on the document provided
- Be concise and direct in your responses
- If information is not in the document, say "This information is not available in the provided document"
- Cite specific sections when relevant
- Maintain a professional and helpful tone"""

def query_document(question: str, document_content: str, api_key: str) -> tuple[str, int]:
    """
    Query the document using OpenAI API with optimized settings.
    Returns: (response_text, tokens_used)
    """
    try:
        client = get_openai_client(api_key)
        
        # Build conversation context for better performance
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(document_content)
            }
        ]
        
        # Add chat history for context (limit to last 5 exchanges for performance)
        recent_history = st.session_state.chat_history[-10:]
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current question
        messages.append({
            "role": "user",
            "content": question
        })
        
        # Make API call with optimized parameters
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster and more cost-effective
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        return answer, tokens_used
    
    except openai.AuthenticationError:
        st.error("❌ Invalid API key. Please check your OpenAI API key.")
        return None, 0
    except openai.RateLimitError:
        st.error("⚠️ Rate limit exceeded. Please wait a moment and try again.")
        return None, 0
    except Exception as e:
        st.error(f"❌ Error querying document: {str(e)}")
        return None, 0

def save_to_memory(role: str, content: str):
    """Save message to session memory."""
    st.session_state.chat_history.append({
        "role": role,
        "content": content
    })

def clear_chat_memory():
    """Clear chat history."""
    st.session_state.chat_history = []
    st.session_state.total_tokens_used = 0

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown('<h1 style="color: #1f77b4; text-align: center;">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.divider()
    
    # API Key Input
    st.subheader("🔑 OpenAI Configuration")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("✅ API key configured")
    
    st.divider()
    
    # Document Upload
    st.subheader("📤 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a TXT file",
        type=["txt"],
        help="Upload a text document to chat with"
    )
    
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        # Load document content with caching
        content = load_document_content(uploaded_file.getvalue(), uploaded_file.name)
        
        if content:
            st.session_state.document_content = content
            st.session_state.document_name = uploaded_file.name
            
            st.success("✅ Document loaded successfully")
            
            # Display document info
            st.markdown('<div class="document-info">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Name", uploaded_file.name[-30:], delta="Loaded")
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            # Character and word count
            char_count = len(content)
            word_count = len(content.split())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Characters", f"{char_count:,}")
            with col2:
                st.metric("Words", f"{word_count:,}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Chat Memory Management
    st.subheader("💾 Memory Management")
    
    if st.session_state.chat_history:
        st.info(f"📊 Chat messages: {len(st.session_state.chat_history)}")
        st.info(f"📈 Total tokens used: {st.session_state.total_tokens_used:,}")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            clear_chat_memory()
            st.success("✅ Chat history cleared")
            st.rerun()
    else:
        st.info("No chat history yet. Start a conversation!")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Advanced Options")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Example Query", use_container_width=True):
            st.session_state.example_query = "Summarize this document in 3 key points"
    
    with col2:
        if st.button("📋 Copy History", use_container_width=True):
            st.info("History copied to clipboard (in production)")


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
def main():
    # Header
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown('<h1 class="header-style">📄 Document Chat Assistant</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subheader-style">Chat with your documents powered by OpenAI</p>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.document_content:
            st.success("✅ Document loaded")
        else:
            st.warning("⚠️ No document loaded")
    
    st.divider()
    
    # Check prerequisites
    if not st.session_state.api_key:
        st.warning("🔑 Please enter your OpenAI API key in the sidebar to get started")
        return
    
    if not st.session_state.document_content:
        st.info("📤 Please upload a TXT file in the sidebar to begin chatting")
        return
    
    # Main chat interface
    st.subheader("💬 Chat with Your Document")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    st.divider()
    
    # Input area
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        user_input = st.chat_input(
            "Ask a question about your document...",
            key="user_input"
        )
    
    with col2:
        submit_button = st.button("Send", use_container_width=True, type="primary")
    
    # Process user input
    if user_input or submit_button:
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Save to memory
            save_to_memory("user", user_input)
            
            # Get response from OpenAI
            with st.chat_message("assistant"):
                with st.spinner("🔄 Processing your query..."):
                    start_time = time.time()
                    
                    response, tokens_used = query_document(
                        user_input,
                        st.session_state.document_content,
                        st.session_state.api_key
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    if response:
                        st.write(response)
                        
                        # Save to memory
                        save_to_memory("assistant", response)
                        
                        # Update token counter
                        st.session_state.total_tokens_used += tokens_used
                        
                        # Display performance metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Response Time", f"{elapsed_time:.2f}s")
                        with metric_col2:
                            st.metric("Tokens Used", tokens_used)
                        with metric_col3:
                            st.metric("Total Tokens", st.session_state.total_tokens_used)
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🚀 Powered by OpenAI GPT-3.5 Turbo")
    with col2:
        st.caption("⚡ Optimized for speed and accuracy")
    with col3:
        st.caption("💾 Context saved in memory")

if __name__ == "__main__":
    main()
