import gradio as gr
from transformers import pipeline

pipe = pipeline("fill-mask", 
                model="google-bert/bert-base-uncased")

output = pipe("The book is [MASK] the table")

for item in output:
    output = item["sequence"]
    print(output)

# the book is on the table
# the book is under the table
# the book is in the table
# the book is off the table
# the book is underneath the table
