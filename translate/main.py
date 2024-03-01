# install DSPy: pip install dspy
import os
from dspy import Example
from setup import create_examples, init_ollama_model, import_json
from dspy.teleprompt import BootstrapFewShot
from similarity_metric import similarity
from translation_module import TranslationModule


ollama_model = init_ollama_model(model='mistral:7b-instruct-v0.2-q6_K', model_type='chat')

json_file = os.path.join(os.path.dirname(__file__), 'translations.json')
translations = import_json(json_file)

examples = create_examples(translations['inputs'],'sanskrit')

teleprompter = BootstrapFewShot(metric=similarity)
compiled_qa = teleprompter.compile(TranslationModule(), trainset=examples)


# Ask any question you like to this simple RAG program.
my_question = "सख्युः सखिसमा वाह्याद्गामियानासनादयः ज्ञातेः स्वसृदुहित्रात्मजाग्रजावरजादयः"
pred = compiled_qa(my_question)


# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.english}")

# not_compiled = TranslationModule()
# print('Uncompiled answer')
# print(not_compiled(my_question).english)

# print(ollama_model.inspect_history(1))
