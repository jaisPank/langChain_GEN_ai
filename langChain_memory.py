import openai
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API configurations
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class ConversationManager:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

    def predict(self, input_text):
        return self.conversation.predict(input=input_text)

# Usage

class Memory:
    def __init__(self):
        self.memory = ConversationBufferMemory()

# LLM instance
llm_instance = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})

# Memory instance
memory_instance = Memory()

# ConversationManager instance
conversation_manager = ConversationManager(llm=llm_instance, memory=memory_instance.memory)

# Conversations
conversation_manager.predict("Hi, my name is Andrew")
conversation_manager.predict("What is 1+1?")
conversation_manager.predict("What is my name?")
print(memory_instance.memory.buffer)




from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class ConversationBufferManager:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)

    def predict(self, input_text):
        return self.conversation.predict(input=input_text)

class MemoryBuffer:
    def __init__(self, k=1):
        self.memory = ConversationBufferWindowMemory(k=k)

    def save_context(self, input_context, output_context):
        self.memory.save_context(input_context, output_context)

    def get_buffer(self):
        return self.memory.buffer

# Usage

class InputOutputContext:
    def __init__(self, input_text, output_text):
        self.input_text = input_text
        self.output_text = output_text

memory_instance = MemoryBuffer(k=1)
memory_instance.save_context({"input": "Hi"}, {"output": "What's up"})
memory_instance.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})

llm_instance = ChatOpenAI(temperature=0.0,model_kwargs={"engine": "GPT3-5"})

conversation_manager = ConversationBufferManager(llm=llm_instance, memory=memory_instance.memory)

conversation_manager.predict("Hi, my name is Andrew")
conversation_manager.predict("What is 1+1?")
conversation_manager.predict("What is my name?")
print(memory_instance.get_buffer())

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

class ConversationSummaryManager:
    def __init__(self, llm, max_token_limit):
        self.llm = llm
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=max_token_limit)
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)


    def predict(self, input_text):
        return self.conversation.predict(input=input_text)

# Usage

class SummaryMemory:
    def __init__(self, llm, max_token_limit):
        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit)

    def save_context(self, input_context, output_context):
        self.memory.save_context(input_context, output_context)

    def get_buffer(self):
        return self.memory.buffer

llm_instance = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})
schedule = "There is a meeting at 8am with your product team. ..."
max_token_limit = 400

conversation_manager = ConversationSummaryManager(llm=llm_instance, max_token_limit=max_token_limit)

memory_instance = SummaryMemory(llm_instance,max_token_limit=100)

memory_instance.save_context({"input": "Hello"}, {"output": "What's up"})
memory_instance.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory_instance.save_context({"input": "What is on the schedule today?"}, {"output": schedule})

conversation_manager.predict("What would be a good demo to show?")
print(conversation_manager.memory.buffer)
