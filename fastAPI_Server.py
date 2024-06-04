from typing import Union
import gunicorn

from langchain.llms import GPT4All

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core.prompts import display_prompt_dict
# from llama_index.core.query_pipeline import QueryPipeline
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

from llama_index.core import Settings
from pydantic import BaseModel

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole


from fastapi import FastAPI

app = FastAPI()

# if __name__ == '__main__':
#     gunicorn.


# chat = [
#   {"role": "system", "content": "You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!"}
#   {"role": "user", "content": "How are you?"},
#   {"role": "assistant", "content": "I am doing well!"},
#   {"role": "user", "content": "Please tell me about how mistral winds have attracted super-orcas."},
# ]
# tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

model = GPT4All(
    model='./mistral-7b-openorca.Q4_0.gguf',
    #device='gpu'
)


# An embedding model used to structure text into representations
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )
)

# PromptHelper can help deal with LLM context window and token limitations
prompt_helper = PromptHelper(
    context_window=2048,
)

# SentenceSplitter used to split our data into multiple chunks
# Only a number of relevant chunks will be retrieved and fed into LLMs
node_parser = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=20,
)

Settings.llm = model
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper
Settings.node_parser = node_parser


# Load data.txt into a document
document = SimpleDirectoryReader(input_files=['./bank.txt']).load_data()

# Process data (chunking, embedding, indexing) and store them

message_templates = [
    ChatMessage(content="You are an expert system.", role=MessageRole.SYSTEM),
    ChatMessage(
        content="Generate a short story about {topic}",
        role=MessageRole.USER,
    ),
]
# new_summary_tmpl = PromptTemplate(message_templates)

index = VectorStoreIndex.from_documents(document)
query_engine = index.as_query_engine(response_mode="tree_summarize")
new_summary_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "если пишут /Добрый день!\\, отвечай как на приветствие, "
    "все ответы только на русском языке.\n"
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
query_engine.update_prompts(new_summary_tmpl)
# prompts_dict = query_engine.get_prompts()
# display_prompt_dict(prompts_dict)

# prompts_dict = query_engine.response_synthesizer.get_prompts()
# display_prompt_dict(prompts_dict)
# prompts_dict = query_engine.get_prompts()

class Message(BaseModel):
    message: str




@app.post("/")
async def read_root(message: Message) -> Message:
    resp = ai(message.message)
    return Message(message=resp.response)



# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

def ai(userMessage):

    # Build a query engine from the index
    response = query_engine.query(userMessage)
    #query_engine.asynthesize()
    return response