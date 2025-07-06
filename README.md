 Custom PDF Chatbot using Gemini Flash 2.0

An intelligent chatbot built with **Streamlit** that responds to user queries based **only on uploaded PDF documents**, using **Google Gemini Flash 2.0 API** and **RAG (Retrieval-Augmented Generation)**. It features a clean chat-style interface (like ChatGPT), dark mode, summary/Q&A highlighting, and prevents hallucinations by grounding all answers in your data.

---

 Features

-  Upload any PDF and chat with it
-  Accurate answers using **Gemini Flash 2.0**
-  Retrieval-Augmented Generation (RAG) with FAISS
-  Clean Chat UI (like WhatsApp/ChatGPT)
-  Dark Mode support
-  Blue question &  green answer bubble styling
-  Clear/Delete chat history
- No hallucinations — answers strictly from your file

---

Tech Stack

| Component       | Tech                      |
|----------------|---------------------------|
| Language        | Python                    |
| UI Framework    | Streamlit                 |
| Vector DB       | FAISS                     |
| LLM             | Google Gemini Flash 2.0   |
| Embeddings      | Google Generative AI      |
| PDF Parser      | PyMuPDF (`fitz`)          |
| Chunking        | LangChain TextSplitter    |


