#from gpt4all import GPT4All
import openai
from langchain.llms import GPT4All

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

from llama_index.core import Settings


#model = GPT4All('/home/rik/Documents/job/chat_gpt/mistral-7b-instruct-v0.1.Q4_0.gguf')
model = GPT4All(model='./mistral-7b-openorca.Q4_0.gguf')

# An embedding model used to structure text into representations
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# PromptHelper can help deal with LLM context window and token limitations
prompt_helper = PromptHelper(context_window=2048)

# SentenceSplitter used to split our data into multiple chunks
# Only a number of relevant chunks will be retrieved and fed into LLMs
node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=20)

Settings.llm = model
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper
Settings.node_parser = node_parser

# Load data.txt into a document
document = SimpleDirectoryReader(input_files=['./text.txt']).load_data()

# Process data (chunking, embedding, indexing) and store them
index = VectorStoreIndex.from_documents(document)

# Build a query engine from the index
query_engine = index.as_query_engine()


response = query_engine.query('Give me my calendar.')
print(response)

# # An embedding model used to structure text into representations
# # embed_model = LangchainEmbedding(
# #     #model
# #     HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# # )
# embed_model = GPT4All("/home/rik/Documents/job/chat_gpt/mistral-7b-instruct-v0.1.Q4_0.gguf")
#
# # PromptHelper can help deal with LLM context window and token limitations
# prompt_helper = PromptHelper(context_window=2048)
#
# # SentenceSplitter used to split our data into multiple chunks
# # Only a number of relevant chunks will be retrieved and fed into LLMs
# node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=20)
#
#
#
# #load
# # Load data.txt into a document
# document = SimpleDirectoryReader(input_files=['./text.txt']).load_data()
#
# # Process data (chunking, embedding, indexing) and store them
# index = VectorStoreIndex.from_documents(document)
#
# # Build a query engine from the index
# query_engine = index.as_query_engine()
#
#
# response = query_engine.query('Give me my calendar.')
# print(response)



# ---------------------------------- HERE -------------------------


# openai.api_base = "http://localhost:4891/v1"
#
# prompt = "The capital of France is "
#
# model = GPT4All("/home/rik/Documents/job/chat_gpt/mistral-7b-instruct-v0.1.Q4_0.gguf")
#
# # response = openai.Completion.create(
# #     model=model,
# #     prompt=prompt,
# #     max_tokens=50,
# #     temperature=0.28,
# #     top_p=0.95,
# #     n=1,
# #     echo=True,
# #     stream=False
# # )
#
# output = model.generate(prompt, max_tokens=50)
#
#
#
# print(output)
