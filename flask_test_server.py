from flask import Flask, request, jsonify

from langchain.llms import GPT4All

from langchain.embeddings import HuggingFaceEmbeddings
# from llama_index.core.query_pipeline import QueryPipeline
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

from llama_index.core import Settings

import asyncio


app = Flask(__name__)

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
query_engine = index.as_query_engine()

# p = QueryPipeline(verbose=True)
# module_dict = {
#     **query_engines,
#     "input": InputComponent(),
#     "summarizer": TreeSummarize(),
#     "join": ArgPackComponent(
#         convert_fn=lambda x: NodeWithScore(node=TextNode(text=str(x)))
#     ),
# }
# p.add_modules(module_dict)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']

    # response = openai.Completion.create(
    #     engine='text-davinci-003',
    #     prompt=user_message,
    #     max_tokens=50,
    #     n=1,
    #     stop=None,
    #     temperature=0.7
    # )

    #assistant_reply = response.choices[0].text.strip()
    resp = ai(user_message)
    # user_message_two = ai("How it going")
    # user_message_three = ai("I like a cakes")

    #print(user_message.response)
    # print(user_message_two.response)
    # print(user_message_three.response)


    return jsonify({'message': resp.response})

def ai(userMessage):

    # Build a query engine from the index
    response = query_engine.query(userMessage)
    #query_engine.asynthesize()
    return response

if __name__ == '__main__':
    app.run(threaded = True)