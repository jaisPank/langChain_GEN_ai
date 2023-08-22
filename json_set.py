import os
from langchain.document_loaders import JSONLoader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai

class ContentProcessor:
    def __init__(self, data_file):
        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")

        self.loader = JSONLoader(file_path=data_file, jq_schema='.messages[].content')
        self.data_content = self.loader.load()

        self.loader = JSONLoader(file_path=data_file, jq_schema='.messages[].summary')
        self.data_summary = self.loader.load()

    def process_content(self):
        content_list = []
        for document in self.data_content:
            content_list.append(document.page_content)
        for document in self.data_summary:
            content_list.append(document.page_content)
        content_set = set(content_list)
        return content_set

    def generate_review(self, content_set):
        review_template = f'You are given a set .Ignore all json tags. Ignore all html tages.Write the possible "model values" and "model trim". Some example are given .From given set. If you find some text likes this :CE 04, C 400 GT, G 310 R, G 310 GS, F 750 GS, F 850 GS, F 850 Adventure, F 900 R, and F 900 XR.Then in CE 04 ,in this CE is "model value" and "model trim" is 04. In C 400  GT , in this C is "model value" and 400 GT is "model trim". In F 850 Adventure , Model values: F and Model Trim : 850 Adventure. : {content_set}.Return the same in table format. "Model value" and "Model trim" as header for table.'
        chat = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})
        response = chat.predict(review_template)
        return response

    def save_to_file(self, content, filename):
        with open(filename, "w") as file:
            file.write(content)

if __name__ == "__main__":
    processor = ContentProcessor(data_file='data.json')
    content_set = processor.process_content()
    response = processor.generate_review(content_set)
    processor.save_to_file(response, "Model_value_model_trim.txt")
