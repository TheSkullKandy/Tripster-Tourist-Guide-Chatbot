import os
import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warning

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Prompt template
def set_custom_prompt():
    template = """
    Use the following context to answer the question.
    If you don't know the answer, say so clearly. Don't make anything up. Stick only to the context.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load LLM
def load_groq_llm():
    return ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0.3,
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1"
    )

# Main app
def main():
    st.set_page_config(page_title="Tourist Guide Chatbot", layout="wide")
    
    # App Header
    st.title("🌎 Tripster Chatbot")
    st.subheader("Explore America and Canada as you Wish!")

    # Sidebar
    with st.sidebar:
        st.header("Settings & Info")
        st.success("Model Used: `llama3-8b-8192`.")
        st.info("Vector DB: FAISS loaded from local disk.")
        st.markdown("**Supported Regions:**\n- 🇺🇸 USA\n- 🇨🇦 Canada")
        st.markdown("---")
        st.markdown("**Built with:**\n- LangChain\n- GROQ API\n- HuggingFace Embeddings\n- Streamlit")

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history display
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your question here...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_groq_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )

            response = qa_chain.invoke({"query": user_input})
            answer = response["result"]
            sources = "\n\n".join(
                [f"📄 `{doc.metadata.get('source', 'Unknown')}`" for doc in response["source_documents"]]
            )

            final_output = f"**Answer:**\n{answer}\n\n---\n**Sources:**\n{sources or 'No sources available.'}"
            st.chat_message("assistant").markdown(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
