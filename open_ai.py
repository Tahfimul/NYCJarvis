from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#import langchain_experimental.agents.create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
#from langchain.agents import create_csv_agent
#from openai import OpenAI
from io import StringIO
import sys
import re
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
import tiktoken
import pandas as pd

# class ChatApp:
#     def __init__(self):
#         # Setting the API key to use the OpenAI API
#         self.client = OpenAI(api_key = "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")
#         self.messages = [
#             {"role": "system", "content": "You are a coding tutor bot to help user write and optimize python code."},
#         ]

#     def chat(self, message):
#         self.messages.append({"role": "user", "content": message})
#         response = self.client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=self.messages
#         )
#         print(response)
#         self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
#         return response.choices[0].message.content

# chatbot = ChatApp()
# print(chatbot.chat("Hello ajlkfsdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"))
# print(chatbot.chat("Hello jkl;afsddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"))
# print(chatbot.chat("Hello afsdjkl;knjkjkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk"))
# print(chatbot.chat("Hello sdjkfllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll"))
# print(chatbot.chat("Hello jksfdaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
# print(chatbot.chat("Hello sfkjdajkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk"))
# print(chatbot.chat("How do you write a function in that language?"))
# print(chatbot.chat("How do I optimize a function in python?"))
# print(chatbot.chat("Give me a detailed response to that"))
# print(chatbot.chat("Give me a long explanation of how python compares to c++"))
# print(chatbot.chat("Give me a long explanation of how python is different than MIPS assembly language"))
# print(chatbot.chat("Give me a long explanation of the difference between a compiled lanugage and an interpreted language"))
# print(chatbot.chat("Give me a long explanation about whether both interpreted language and compiled languages are platform dependent"))
# print(chatbot.chat("Give me a long winded history about python"))
# print(chatbot.chat("can we determine, in general, whether a given algorithm will eventually halt or contnue running indefinitely?"))
# print(chatbot.chat("what does it mean for a computational system to be turing complete, and how does this concept relate to the capabilities of different programming languages and machines?"))
# print(chatbot.chat("how do the principles of quantum mechanics enable the development of quantum computers, and what potential advantages do they have over classical computers for certain types of problems?"))

class ChatApp2:
    def __init__(self, paths):
        # Setting the API key to use the OpenAI API
        self.chatOpenAI = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key="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")
   
        # self.csv_agent = create_csv_agent(
        #             ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key="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"),
        #             paths,
        #             verbose=True
        #     )
        df_list = []
        chunksize = 400000
        with pd.read_csv(paths, chunksize=chunksize) as reader:
            for chunk in reader:
                df_list.append(chunk)

        df_res = pd.concat(df_list)        

        self.agent = create_pandas_dataframe_agent(
                        ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key="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"),
                        df_res,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
        )    
        # self.datasetsAgent = create_csv_agent( self.chatOpenAI, "datasets.csv", verbose=True,
        #             agent_type=AgentType.OPENAI_FUNCTIONS, low_memory=False,
        #               )              

    def chat(self, message):
        output_capture = StringIO()
        sys.stdout = output_capture
        with get_openai_callback() as cb:
            # self.csv_agent.run(message)
            self.agent.run(message)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost ${cb.total_cost}")
        sys.stdout = sys.__stdout__
        output = output_capture.getvalue()
        cleaned_output = re.sub(r'\x1b[^m]*m', '', output)
        return cleaned_output
        return output
    def queryChat(self, messages):
        return self.chatOpenAI(messages)    

#message = "which name has the highest vitality?"
# message = "HOW MANY ROWS ARE there?"
# message = "how many tokens are in this entire query?"
# message = "which List Agency Desc shows up the most?"
# message = "what is the highest job#?"

# message = "who is most likely to get killed in a car crash?"

# message = "which rows are related to the following question: 'which school had the highest sat score?'?" 

message = "how many accidents happened in june?"

datasets = 'datasets/Motor_Vehicle_Collisions_-_Crashes_20231125.csv'



messages = [
    # SystemMessage(
    #     content="You are a helpful assistant that translates English to French."
    # ),
    HumanMessage(
        content = "Given the datasets ['cars', 'spiderman', 'dora', 'donald duck', 'character', 'plane']"     
        ),
    HumanMessage(
        content="what did I say about trees again?"
    ) 
]

app2 = ChatApp2(datasets)    
print(app2.chat(message))
# print(app2.queryChat(messages))

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

f = open("datasets.csv", "r")
lines = f.read()
print(num_tokens_from_string(lines, "cl100k_base"))

f.close()

# 1. Break each dataset into chunks of ~1500 tokens files
# 2. Create a map of indices from each dataset title to indices corresponding to each dataset title's indices in the list of csv agents like this:
#     {<dataset title's index in dataset titles list>:[<dataset title's index1 in list of csv agents>, <dataset title's index2 in list of csv agents>...]}

# 3. Create list of csv agents using the files in the datasets folder
# 4. Get user input question. Limit question token size to 1000
# 5. 

