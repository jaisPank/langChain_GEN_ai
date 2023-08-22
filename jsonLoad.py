import openai
from pprint import pprint
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import JSONLoader
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API configurations
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["summary"] = record.get("summary")
    metadata["endDateUtc"] = record.get("endDateUtc")

    return metadata


loader = JSONLoader(
    file_path='data.json',
    jq_schema='.messages[]',
    content_key="content",
    metadata_func=metadata_func
)

data = loader.load()
pprint(data)  ##Print in proper format, so not using print()

