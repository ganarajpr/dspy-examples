# Translation from Sanskrit to English using DSPY

## Introduction
This is a simple DSPy module that implements a few-shot translation model from Sanskrit to English. The model is based on the [Ollama](https://github.com/shubham1710/ollama) model. I am using `mistral` as the LLM.


## Installation and Usage
1. Install DSPy: `pip install -r requirements.txt`
2. Ensure you have ollama installed. 
3. In your command prompt run `ollama pull mistral:7b-instruct-v0.2-q6_K`
4. Go to `similarity.py` uncomment the print lines and run it.
```
sentence1 = "This is a sample sentence."
sentence2 = "Here is an example sentence."
similarity = sentence_similarity(sentence1, sentence2)

print(f"Similarity between the sentences: {similarity}")
``` 
5. Finally run `python main.py`
6. Profit

## Explanation
The `TranslationModule` is a simple module that uses the `mistral` LLM to translate Sanskrit to English. The module is trained on a few-shot learning task using the `BootstrapFewShot` teleprompter. We are using `BAII/bge-m3`  model to do similarity search which is used as our metric to assess whether the model is doing the right thing or not. You can look it up on [HuggingFace](https://huggingface.co/BAAI/bge-m3). 
