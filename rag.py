import os
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

## env time
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv("groq_api")
## DOCS_folder="./docs/"
FAISS_Index="./fais_inddex"

## 1 Load PDF
print("Loading Loading the PDFssss")
loader = PyMuPDFLoader("Detection_of_Fake_Accounts_on_Social_Media_Using_Multimodal_Data_With_Deep_Learning.pdf")
## 2 now convert to doc
docs = loader.load()

##3 chunk chunk chunk
print("Start tthe Chunking")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks=splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

## 4 Embed
print("Now embedding time")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)

## 5 now will store in the vectordatabase
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(FAISS_Index)

## 6 now will retrive
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4} ## top 4 relevant chunks
)

## 7 LLM call
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0, ## deterministic, good for Q&A
)

## 8 Now retrival Q&A
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

## 9 Question time
def ask(question: str):
    print(f"\nQ: {question}")
    result = chain.invoke(question)
    print(f"\nA: {result}")
    print("\n---- Sources ----")
    for doc in retriever.invoke(question):
        print(f" {doc.metadata.get('source', 'unknown')} | Page {doc.metadata.get('page', '?')}")

ask('What is meant by fake accounts on social media?')