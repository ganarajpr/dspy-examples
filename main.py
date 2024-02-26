# install DSPy: pip install dspy
import dspy
from dspy.teleprompt import LabeledFewShot
import json
from dspy import Example

ollama_model = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='mistral:7b-instruct-v0.2-q6_K', stop='\n\n', model_type='chat')
# ollama_model = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='gemma', stop='\n\n', model_type='chat')

# This sets the language model for DSPy.
dspy.settings.configure(lm=ollama_model)


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
my_question = "99+1021"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_qa(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

uncompiled_math = MathModule()
print('Uncompiled answer')
print(uncompiled_math(my_question).answer)

print(ollama_model.inspect_history(1))


# [Example({'question': 'What is 193+934', 'answer': '1127'}) (input_keys={'question'}), Example({'question': 'What is 565+916', 'answer': '1481'}) (input_keys={'question'}), Example({'question': 'What is 80+920', 'answer': '1000'}) (input_keys={'question'}), Example({'question': 'What is 528+254', 'answer': '782'}) (input_keys={'question'}), Example({'question': 'What is 674+840', 'answer': '1514'}) (input_keys={'question'}), Example({'question': 'What is 788+330', 'answer': '1118'}) (input_keys={'question'}), Example({'question': 'What is 520+735', 'answer': '1255'}) (input_keys={'question'}), Example({'question': 'What is 630+852', 'answer': '1482'}) (input_keys={'question'}), Example({'question': 'What is 835+76', 'answer': '911'}) (input_keys={'question'}), Example({'question': 'What is 27+128', 'answer': '155'}) (input_keys={'question'})]
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 564.72it/s]
# Bootstrapped 1 full traces after 10 examples in round 0.
# generate_answer = ChainOfThought(Signature(question -> answer
#     instructions='Answer questions about basic arithmetic'
#     question = Field(annotation=str required=True json_schema_extra={'desc': 'a question about arithmetic', '__dspy_field_type': 'input', 'prefix': 'Question:'})
#     answer = Field(annotation=str required=True json_schema_extra={'desc': 'one number', '__dspy_field_type': 'output', 'prefix': 'Answer:'})
# ))
# Question: 99+1234
# Predicted Answer: 1333
# Uncompiled answer
# 1021
