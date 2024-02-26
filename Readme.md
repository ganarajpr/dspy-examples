### DSPy for Math

A simple Dspy program which uses the DSPy library to do some math. 

Assuming that we have a bunch of math related questions that we wish to ask LLM,
we create a DSPy program that will do the math for us.

In this experiment, we use Chain of Thought to ask the question "What is 1 + 1?"

We train the DSPy program with the following examples to Bootstrap it:

```js
{
  "inputs": [
    {
      "question": "What is 193+934",
      "answer": "1127"
    },
    {
      "question": "What is 565+916",
      "answer": "1481"
    },
    {
      "question": "What is 80+920",
      "answer": "1000"
    },
    {
      "question": "What is 528+254",
      "answer": "782"
    },
    {
      "question": "What is 674+840",
      "answer": "1514"
    },
    {
      "question": "What is 788+330",
      "answer": "1118"
    },
    {
      "question": "What is 520+735",
      "answer": "1255"
    },
    {
      "question": "What is 630+852",
      "answer": "1482"
    },
    {
      "question": "What is 835+76",
      "answer": "911"
    },
    {
      "question": "What is 27+128",
      "answer": "155"
    }
  ]
}
```
