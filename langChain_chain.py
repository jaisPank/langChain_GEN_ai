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
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class LLMChainClass:
    def __init__(self, temperature, model_kwargs, prompt_template):
        self.llm = ChatOpenAI(temperature=temperature, model_kwargs=model_kwargs)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, product):
        return self.chain.run(product)

# Usage

class_name = LLMChainClass(
    temperature=0.0,
    model_kwargs={"engine": "GPT3-5"},
    prompt_template="What is the best name to describe a company that makes {product}?"
)

product_name = "Queen Size Sheet Set"
result = class_name.run(product_name)
print(result)


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

class ChainManager:
    def __init__(self, temperature, model_kwargs):
        self.llm = ChatOpenAI(temperature=temperature, model_kwargs=model_kwargs)
        self.chains = []

    def add_chain(self, prompt_template):
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        self.chains.append(chain)

    def create_and_add_prompt(self, prompt_template):
        return ChatPromptTemplate.from_template(prompt_template)

    def run_chains(self, product_name):
        results = []
        for chain in self.chains:
            result = chain.run(product_name)
            results.append(result)
        return results

# Usage

class_name = ChainManager(temperature=0.9, model_kwargs={"engine": "GPT3-5"})

# Prompt templates
first_prompt_template = "What is the best name to describe a company that makes {product}?"
second_prompt_template = "Write a 20 words description for the following company: {company_name}"

# Create and add prompts
first_prompt = class_name.create_and_add_prompt(first_prompt_template)
second_prompt = class_name.create_and_add_prompt(second_prompt_template)

# Add chains
class_name.add_chain(first_prompt)
class_name.add_chain(second_prompt)

# Create a SimpleSequentialChain
overall_simple_chain = SimpleSequentialChain(chains=class_name.chains, verbose=True)

# Run the SimpleSequentialChain
product_name = "Queen Size Sheet Set"
results = overall_simple_chain.run(product_name)
print(results)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain

class ReviewProcessingChain:
    def __init__(self, temperature):
        self.llm = ChatOpenAI(temperature=temperature,model_kwargs={"engine":"GPT3-5"})
        self.chains = []

    def add_llm_chain(self, prompt_template, output_key):
        chain = LLMChain(llm=self.llm, prompt=prompt_template, output_key=output_key)
        self.chains.append(chain)

    def create_and_add_prompt(self, prompt_template):
        return ChatPromptTemplate.from_template(prompt_template)

    def process_review(self, review):
        results = self.chains[0](Review=review)
        for chain in self.chains[1:]:
            results = chain(**results)
        return results

# Usage
class_name = ReviewProcessingChain(temperature=0.9)

# Prompt templates
first_prompt_template = "Translate the following review to english:\n\n{Review}"
second_prompt_template = "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
third_prompt_template = "What language is the following review:\n\n{Review}"
fourth_prompt_template = "Write a follow up response to the following summary in the specified language:\n\nSummary: {summary}\n\nLanguage: {language}"

# Create and add prompts
first_prompt = class_name.create_and_add_prompt(first_prompt_template)
second_prompt = class_name.create_and_add_prompt(second_prompt_template)
third_prompt = class_name.create_and_add_prompt(third_prompt_template)
fourth_prompt = class_name.create_and_add_prompt(fourth_prompt_template)
# Add LLM chains
class_name.add_llm_chain(first_prompt, output_key="English_Review")
class_name.add_llm_chain(second_prompt, output_key="summary")
class_name.add_llm_chain(third_prompt, output_key="language")
class_name.add_llm_chain(fourth_prompt, output_key="followup_message")
# Create a SequentialChain
input_variables = ["Review"]
output_variables = ["English_Review", "summary", "followup_message"]
overall_chain = SequentialChain(chains=class_name.chains, input_variables=input_variables, output_variables=output_variables, verbose=True)

review= "Queen Size Sheet Set"
results = overall_chain(review)
print(results)
