import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", 
                model="google/gemma-3-1b-it")

def response(message, history):
    response = pipe(message)

    print(response)

    for item in response:
        output = item["generated_text"]

    return output 

gr.ChatInterface(fn=response).launch()