import dspy
from translation_signature import Translation


class TranslationModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_answer = dspy.Predict(Translation)
    
    def forward(self, sanskrit):
        prediction = self.generate_answer(sanskrit=sanskrit)
        return dspy.Prediction(english=prediction.english)
