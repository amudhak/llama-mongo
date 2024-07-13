# llama-mongo
Process data using Llama3 and then store this data in MongoDB, while integrating Atlas Vector Search with LangChain.

## Install dependencies and set up
- Create a new Atlas Cluster
- Run the following commands:
'''cmd
python3 -m pip install "pymongo[srv]"
pip install --quiet langchain langchain-openai langchain-mongodb langchain-community pymongo pypdf
pip install -U sentence-transformers
pip install tf-keras
pip install -U langchain-huggingface
ollama pull llama3
'''
- Then add the following to the top of your python file:
'''python
import getpass, os, pymongo, pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
'''
- Define your Atlas cluster's SRV connection string (following format: mongodb+srv://<username>:<password>@<clusterName>.<hostname>.mongodb.net)
- Following this tutorial:
  - https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/
