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
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class Model:
    def __init__(self, temperature, model_kwargs):
        self.chat = ChatOpenAI(temperature=temperature, model_kwargs=model_kwargs)

class Prompt:
    def __init__(self, template_string, style, text):
        self.prompt_template = ChatPromptTemplate.from_template(template_string)
        self.messages = self.prompt_template.format_messages(style=style, text=text)

class OutputParser:
    def __init__(self, response_schemas,template_string,text):
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.format_instruction = self.output_parser.get_format_instructions()
        self.prompt_template = ChatPromptTemplate.from_template(template_string)
        self.messages = self.prompt_template.format_messages(format_instructions=self.format_instruction, text=text)

    def parse_response(self, response_content):
        output_dict = self.output_parser.parse(response_content)
        return output_dict

# Usage

# Model class
model = Model(temperature=0.0, model_kwargs={"engine": "GPT3-5"})


# Prompt class for customer

#Template String
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
customer_style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
customer_prompt = Prompt(template_string, customer_style, customer_email)

# Call the LLM to translate to the style of the customer message
customer_response = model.chat(customer_prompt.messages)
print("Output For Respectful Tone")
print(customer_response.content)
print()



# Prompt class for service

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""
service_prompt = Prompt(template_string, service_style_pirate, service_reply)

# Call the LLM with service prompt
service_response = model.chat(service_prompt.messages)
print("Output for English Pirate")
print(service_response.content)
print()



# OutputParser class

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]




customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

output_parser = OutputParser(response_schemas,template_string=review_template_2,text=customer_review)
output_parser_response = model.chat(output_parser.messages)
# Parse output_parser_response response
customer_output_dict = output_parser.parse_response(output_parser_response.content)
print(type(customer_output_dict))
print(customer_output_dict)