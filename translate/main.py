# install DSPy: pip install dspy
import os
from setup import create_examples, init_ollama_model, import_json
from dspy.teleprompt import BootstrapFewShot, SignatureOptimizer
from similarity_metric import similarity
from translation_module import TranslationModule


ollama_model = init_ollama_model(model='mistral:7b-instruct-v0.2-q6_K', model_type='chat')




def get_examples(json_file): 
    json_file = os.path.join(os.path.dirname(__file__), json_file)
    translations = import_json(json_file)
    keys = set(translations['inputs'][0].keys())
    keys.remove('english')
    examples = create_examples(translations['inputs'], keys)
    return examples



sanskrit_examples = get_examples('translations.json')

teleprompter = BootstrapFewShot(metric=similarity)
compiled_qa = teleprompter.compile(TranslationModule(), trainset=sanskrit_examples)

kwargs = dict(num_threads=1, display_progress=True, display_table=0)
optimiser = SignatureOptimizer(metric=similarity, prompt_model=ollama_model)
compiled_prompt_opt = optimiser.compile(compiled_qa.deepcopy(), devset=sanskrit_examples, eval_kwargs=kwargs)

# Ask any question you like to this simple RAG program.
my_question = "सख्युः सखिसमा वाह्याद्गामियानासनादयः ज्ञातेः स्वसृदुहित्रात्मजाग्रजावरजादयः"
pred = compiled_qa(original=my_question, lang='sanskrit')
optimised_pred = compiled_prompt_opt(original=my_question, lang='sanskrit')

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.english}")
print(f"Predicted Answer: {optimised_pred.english}")


# kannada_examples = get_examples('en-kn-dspy.json')

# teleprompter = BootstrapFewShot(metric=similarity)
# compiled_qa = teleprompter.compile(TranslationModule(), trainset=kannada_examples)

# # Ask any question you like to this simple RAG program.
# my_question = "ಸತ್ತ ಪತಿಯ ಮೋಸದಾಟ ಗೊತ್ತಾದ ಪತ್ನಿ ಶಾಕ್, ಸತ್ತಾಗಿದೆ, ಮಾಡೋದಿನ್ನೇನು?"
# pred = compiled_qa(original=my_question, lang='kannada')

# # Print the contexts and the answer.
# print(f"Question: {my_question}")
# print(f"Predicted Answer: {pred.english}")



# not_compiled = TranslationModule()
# print('Uncompiled answer')
# print(not_compiled(my_question).english)

print(ollama_model.inspect_history(1))
