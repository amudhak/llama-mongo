import pprint, requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define a prompt template
template = """

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = ChatOllama(model="llama3", format="json")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

embedding = FastEmbedEmbeddings()

# Connect to your Atlas cluster
def connect_to_mongodb(atlas_connection_string):
    client = MongoClient(atlas_connection_string)
    return client

# Define collection and index name
def get_database(client):
    db = client.langchain_db
    db_name = "langchain_db"
    collection = db.test
    collection_name = "test"
    return client[db_name][collection_name]

# Load the PDF
def load_pdf(pdf_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    # Make the request
    response = requests.get(pdf_url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the PDF to a local file
        with open("downloaded_pdf.pdf", "wb") as f:
            f.write(response.content)
        # Load the PDF from the local file
        loader = PyPDFLoader("downloaded_pdf.pdf")
        return loader.load()
    else:
        print(f"Failed to download the PDF. Status code: {response.status_code}")

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

# Usage
ATLAS_CONNECTION_STRING = "mongodb+srv://amudhakrishnan:N88TxJg5jyUN@cluster0.jo75rgz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = connect_to_mongodb(ATLAS_CONNECTION_STRING)

atlas_collection = get_database(client)
vector_search_index = "vector_index"

# Change pdf url here
PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6494975/pdf/main"
data = load_pdf(PDF_URL)

# Split PDF into documents
docs = text_splitter.split_documents(data)

# Create the vector store
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents = docs,
    embedding = embedding,
    collection = atlas_collection,
    index_name = vector_search_index
)

# Instantiate Atlas Vector Search as a retriever
retriever = vector_store.as_retriever(
   search_type = "similarity",
   search_kwargs = { "k": 10 }
)

# Construct a chain to answer questions on your data
rag_chain = (
   { "context": retriever | format_docs, "question": RunnablePassthrough()}
   | custom_rag_prompt
   | llm
   | StrOutputParser()
)

print("hello")

# Prompt the chain
question = "What is the risk of subsequent primary neoplasms in survivors of adolescent and young adult cancer?"
answer = rag_chain.invoke(question)

print("Question: " + question)
print("Answer: " + answer)

# Return source documents
documents = retriever.get_relevant_documents(question)
print("\nSource documents:")
pprint.pprint(documents)