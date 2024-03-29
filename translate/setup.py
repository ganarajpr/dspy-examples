import dspy
import json
from dspy import Example

def init_ollama_model(model, model_type):
    ollama_model = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', 
                               model=model, stop='\n\n', model_type=model_type)
    dspy.settings.configure(lm=ollama_model)
    return ollama_model


def import_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def create_examples(data, inputs):
    output = []
    for example in data:
        output.append(Example(base=example).with_inputs(*inputs))
    return output
    

# Given the following attributes of a product
# 47,44,45,40,38,L,One size,W12,W3,W9.5,XL,XXL
# which belong to the category
# Safety > Personal Protection & Clothing > Clothing & Workwear > Arc Flash, Heat Resistant, & Welding ClothingSafety Overshoes

#Can you provide these attributes in a logical order . Respond back with only the attributes
