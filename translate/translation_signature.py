import dspy

class Translation(dspy.Signature):
    """Translate to English"""

    sanskrit = dspy.InputField(desc="a verse in sanskrit")
    english = dspy.OutputField(desc="translation in english")
