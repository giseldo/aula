import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", 
                model="google/gemma-3-1b-it",
                num_return_sequences=1,
                device_map="auto")

def response(message, history):
    messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": message},]
        },
    ],
    ]

    response = pipe(messages)

    print(response)
    
    return response[0][0]['generated_text'][2]['content']

gr.ChatInterface(fn=response).launch()
