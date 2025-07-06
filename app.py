import os
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# ====== CONFIG ======
import os
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="📄 PDF Chatbot", layout="centered")

# ====== Custom UI Styling ======
st.markdown("""
    <style>
    .message-container {
        max-height: 65vh;
        overflow-y: auto;
        padding: 0.5rem 1rem;
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .message {
        padding: 10px 15px;
        border-radius: 20px;
        margin-bottom: 10px;
        max-width: 75%;
        line-height: 1.4;
        word-wrap: break-word;
    }
    .user {
        background-color: #DCF8C6;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot {
        background-color: #F1F0F0;
        align-self: flex-start;
        margin-right: auto;
    }
    .chat-title {
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: #1877f2;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='chat-title'>📄 PDF Chatbot</div>", unsafe_allow_html=True)

# ====== Session State ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "full_pdf_text" not in st.session_state:
    st.session_state.full_pdf_text = ""

# ====== PDF Upload & Processing ======
uploaded_file = st.file_uploader("📤 Upload a PDF to start chatting", type="pdf")

def extract_text(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def create_qa_chain(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        system_message="""
You are a PDF Expert Assistant. Your job is to read, understand, and answer user questions *strictly* based on the content of the uploaded PDF.

🎯 Role: You are a calm, professional, and precise assistant.
🧠 Language: Always answer in **clear, fluent English**.
📌 Task: Answer questions using only the PDF. Be concise but informative.
❌ Constraints:
- Do not guess.
- If the PDF lacks info, say: "The document does not contain enough information to answer that."
- Do not generate opinions or outside facts.

Structure your answers in short paragraphs or bullet points if needed.
""".strip()
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )
    return chain

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("🔍 Reading and indexing your PDF..."):
        text = extract_text(uploaded_file)
        st.session_state.full_pdf_text = text  # ✅ Store text for summary
        st.session_state.qa_chain = create_qa_chain(text)
    st.success("✅ PDF processed! Start chatting below.")

# ====== Chat Message Display ======
st.markdown("<div class='message-container'>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "bot"
    st.markdown(f"<div class='message {role}'>{msg.content}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ====== Chat Input Handling ======
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask a question based on your PDF...")
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        result = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append(AIMessage(content=result))
        st.rerun()

# ====== Sidebar Options ======
with st.sidebar:
    st.title("⚙️ Options")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.full_pdf_text = ""
        st.rerun()

    if st.button("📋 Summarize PDF"):
        if st.session_state.full_pdf_text:
            with st.spinner("Generating summary..."):
                summary_prompt = "Give a clear and concise summary of this PDF document."
                summary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
                summary = summary_llm.invoke(summary_prompt + "\n\n" + st.session_state.full_pdf_text[:20000])
                st.session_state.chat_history.append(HumanMessage(content="📋 Summarize PDF"))
                st.session_state.chat_history.append(AIMessage(content=summary.content))
                st.rerun()
        else:
            st.warning("⚠️ Please upload and process a PDF first.")
