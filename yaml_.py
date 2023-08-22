import openai
from pprint import pprint
import os
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API configurations
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

with open("data.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0,model_kwargs={"engine":"GPT3-5"}),
    toolkit=json_toolkit,
    verbose=True
)

out=json_agent_executor.run("You are given a a json. Given json contain a key 'content'. Return all 'content' values.")
print("From Json Values of Summary :", out)
review_template = f'You are given a text .Write the possible model values and model trim. Some example are given .Example 1 :CE 04 ,in this CE is model value and model trim is 04. Example 2 : C 400  GT , in this C is model value and 400 GT is model trim. : {out}. You have give in dict model value and model.'
prompt_template = ChatPromptTemplate.from_template(review_template)
messages = prompt_template.format_messages(text=out)
chat = ChatOpenAI(temperature=0.0, engine="GPT3-5")
response = chat(messages)
print(response.content.strip())