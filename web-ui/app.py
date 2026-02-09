import streamlit as st
import httpx
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
APP_API_URL = os.getenv("APP_API_URL", "http://localhost:8000/query")

st.set_page_config(
    page_title="Cosense RAG Chat",
    page_icon="📝",
    layout="wide"
)

def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def sidebar():
    """Render the sidebar with settings."""
    st.sidebar.title("🛠️ Settings")
    
    top_k = st.sidebar.slider("Number of retrieved documents (top_k)", 1, 10, 5)
    
    st.sidebar.divider()
    st.sidebar.info(
        "このシステムはScrapbox（Cosense）のデータをソースとし、"
        "SPLADEによる疎ベクトル検索とGemma 3を活用したRAGシステムです。"
    )
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    return top_k

def main():
    init_session_state()
    top_k = sidebar()

    st.title("📝 Cosense RAG with local Gemma 3")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 参考にしたScrapboxページ"):
                    for src in message["sources"]:
                        st.write(f"- [{src['title']}]({src['url']}) (Score: {src.get('score', 0):.2f})")

    # Chat input
    if prompt := st.chat_input("Scrapboxについて質問してください"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            sources = []
            
            try:
                # Call App API with streaming
                with httpx.stream(
                    "POST", 
                    APP_API_URL, 
                    json={"user_query": prompt, "top_k": top_k}, 
                    timeout=None
                ) as response:
                    if response.status_code != 200:
                        st.error(f"API Error: {response.status_code}")
                    else:
                        buffer = ""
                        metadata_received = False
                        
                        for chunk in response.iter_text():
                            buffer += chunk
                            
                            # Try to extract metadata if not yet received
                            if not metadata_received and "\n---\n" in buffer:
                                parts = buffer.split("\n---\n", 1)
                                try:
                                    metadata_str = parts[0]
                                    metadata = json.loads(metadata_str)
                                    if metadata.get("type") == "metadata":
                                        sources = metadata.get("sources", [])
                                        metadata_received = True
                                        buffer = parts[1] # Keep the rest of the text
                                except json.JSONDecodeError:
                                    pass
                            
                            if metadata_received:
                                full_response = buffer
                                response_placeholder.markdown(full_response + "▌")
                        
                        # Final update
                        response_placeholder.markdown(full_response)
                        
                        # Display sources in an expander
                        if sources:
                            with st.expander("📚 参考にしたScrapboxページ"):
                                for src in sources:
                                    st.write(f"- [{src['title']}]({src['url']}) (Score: {src.get('score', 0):.2f})")
                
                # Store assistant message in history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Error connecting to App API: {e}")

if __name__ == "__main__":
    main()
