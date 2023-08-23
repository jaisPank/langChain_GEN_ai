import openai
import os
from dotenv import load_dotenv

class ChatAgent:
    def __init__(self):
        load_dotenv()
        
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        
        from langchain.agents import load_tools, initialize_agent
        from langchain.agents import AgentType

        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})
        self.tools = load_tools(["llm-math", "wikipedia"], llm=llm)
        
        self.agent = initialize_agent(
            self.tools,
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def ask_question(self, question):
        result = self.agent(question)
        return result

# Create an instance of the ChatAgent class
chat_agent = ChatAgent()

# Example usage
question1 = "What is the 25% of 300?"
result1 = chat_agent.ask_question(question1)
print(result1)

question2 = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"

result2 = chat_agent.ask_question(question2)
print(result2)
