import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", 
                model="openai-community/gpt2",
                num_return_sequences=1,
                max_length=10)

def response(message, history):
    response = pipe(message)
    for item in response:
        output = item["generated_text"]
    
    return output 

gr.ChatInterface(fn=response).launch()
