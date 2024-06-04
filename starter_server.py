from flask import Flask, request, jsonify
from gpt4all import GPT4All

from transformers import (GPT2LMHeadModel, CLIPTokenizer)
# GPT2TokenizerFast)

app = Flask(__name__)
print(app)
model = GPT4All("/home/rik/Documents/job/chat_gpt/mistral-7b-instruct-v0.1.Q4_0.gguf", device='gpu')
model.eval() @app.route('/generate', methods=['POST'])

def generate():
    data = request.get_json()
    prompt = data['prompt']
    length = data['length']
    # input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # output = model.generate(input_ids, max_length=length, do_sample=True)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    output = model.generate("The capital of France is ", max_tokens=50)

    return jsonify({'response': output})
