import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Optional: silence tokenizer fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Step 1: Load GROQ-compatible OpenAI LLM
def load_llm():
    return ChatOpenAI(
        model="llama3-8b-8192",  # ✅ Use updated GROQ-supported model
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=512
    )

# Step 2: Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the following context to answer the user's question.
If you don't know the answer, just say you don't know — don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

def set_custom_prompt(prompt_template):
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Step 3: Load FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Run it
if __name__ == "__main__":
    user_query = input("Write your query: ")
    response = qa_chain.invoke({'query': user_query})
    print("\n💬 RESPONSE:\n", response["result"])
    print("\n📄 SOURCE DOCUMENTS:\n", response["source_documents"])
