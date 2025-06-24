"""
Description (react_agent_scratch.py):
This script implements a foundational agent based on the ReAct (Reasoning and Acting) paradigm. 
The Agent class handles interaction with an OpenAI LLM, maintaining a message history. 
The LLM is guided by a specific prompt that instructs it to operate in a "Thought, Action, PAUSE, Observation" loop. 
Python functions like calculate and average_dog_weight serve as the "Actions" (tools) available to the agent. 
The query function orchestrates the entire process: it sends user questions to the agent, parses the LLM's output for actions, executes these actions using the defined Python functions, and feeds the results (observations) back to the LLM until a final answer is produced or a maximum number of turns is reached. 
This script lays the groundwork for understanding agent mechanics before introducing more advanced frameworks.
"""

import openai
import re
import httpx # Not explicitly used in the final agent, but often useful for API calls
import os
from dotenv import load_dotenv

_ = load_dotenv()

client = openai.OpenAI() # Assuming OPENAI_API_KEY is set in .env

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name == "Scottish Terrier": # Exact match is better for tool use
        return("Scottish Terriers average 20 lbs")
    elif name == "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name == "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs") # Default for unknown breeds

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

action_re = re.compile('^Action: (\w+): (.*)$') # python regular expression to select action

def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(f"\nLLM Output {i}:\n{result}")

        if "Answer:" in result: # Check if the agent has an answer
            return result # Or just print and return

        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]

        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(f" -- Running Action: {action} {action_input}")
            observation = known_actions[action](action_input)
            print(f"Observation: {observation}")
            next_prompt = "Observation: {}".format(observation)
        else:
            # No action and no answer, something might be off or it's the end of a thought process
            print("No action found, stopping.")
            return result # Return the last thought
    print("Max turns reached.")
    return bot.messages[-1]["content"] # Return the last assistant message

if __name__ == "__main__":
    print("\n--- Running query for combined dog weight ---")
    question_combined = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
    final_result_combined = query(question_combined)
    print(f"\nFinal Result for combined weight:\n{final_result_combined}")

    print("\n--- Running query for a calculation ---")
    question_calc = "What is 100 / 5 + 3?"
    final_result_calc = query(question_calc)
    print(f"\nFinal Result for calculation:\n{final_result_calc}")