# install DSPy: pip install dspy
import dspy
from dspy.teleprompt import LabeledFewShot
import json
from dspy import Example
import uuid
# Ollam is now compatible with OpenAI APIs
# 
# To get this to work you must include `model_type='chat'` in the `dspy.OpenAI` call. 
# If you do not include this you will get an error. 
# 
# I have also found that `stop='\n\n'` is required to get the model to stop generating text after the ansewr is complete. 
# At least with mistral.

ollama_model = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='mistral:7b-instruct-v0.2-q6_K', stop='\n\n', model_type='chat')

# This sets the language model for DSPy.
dspy.settings.configure(lm=ollama_model)

# This is not required but it helps to understand what is happening

with open('questions.json') as f:
    questions = json.load(f)

output = []
for example in questions['inputs']:
    output.append(Example(base=example).with_inputs('question'))

print(output)

# This is the signature for the predictor. It is a simple question and answer model.
class MathQA(dspy.Signature):
    """Answer questions about basic arithmetic"""

    question = dspy.InputField(desc="a question about arithmetic")
    answer = dspy.OutputField(desc="one number")


class MathModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_answer = dspy.ChainOfThought(MathQA)
    
    def forward(self, question):
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)



from dspy.evaluate import answer_exact_match
from dspy.teleprompt import BootstrapFewShot

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=answer_exact_match)
# evaluate(MathModule(), metric=llm_metric)

# Compile!
compiled_qa = teleprompter.compile(MathModule(), trainset=output)
print(compiled_qa)


# Ask any question you like to this simple RAG program.
my_question = "99+1234"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_qa(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

uncompiled_math = MathModule()
print('Uncompiled answer')
print(uncompiled_math(my_question).answer)
