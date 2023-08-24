from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import openai
import os

class ConversationProcessor:
    def __init__(self):
        
        load_dotenv()

        # Set OpenAI API configurations
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")

        # Load documents from text files
        self.documents = self.read_text_files()

        # Get the text chunks
        self.document_chunks = self.get_chunks(self.documents)

        # Create vector store
        self.vectorstore = self.get_vectorstore(self.document_chunks)

        # Create conversation chain
        self.conversation_chain = self.get_chain(self.vectorstore)

    def read_text_files(self):
        # List of file paths you want to read
        loader = DirectoryLoader('./text_file/', glob="./*.txt", loader_cls=TextLoader)
        documents = loader.load()
        return documents

    def get_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(documents)
        return document_chunks

    def get_vectorstore(self, document_chunks):
        
        embeddings = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-base-v2")

        # Getting vectore store from db if exists.
        vectorstoreOld = FAISS.load_local("faiss_index", embeddings)
        if vectorstoreOld:
            return vectorstoreOld
        else:
            
            vectorstore = FAISS.from_documents(documents=document_chunks, embedding=embeddings)

            #Saving vectore store to Directory .
            vectorstore.save_local("faiss_index")
            return vectorstore

    def get_chain(self, vectorstore):
        llm = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})

        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        return conversation_chain

    def process_query(self, query):
        response = self.conversation_chain(query)
        return response
    
    def process_llm_response(self,response):
        print(response['result'])
        print('\n\nSources:')
        for source in response["source_documents"]:
            print(source.metadata['source'])

if __name__ == "__main__":
    processor = ConversationProcessor()
    template_query = "You have given a documents of type agreement.There are different type of agreement. So from these document get Start date and end date of agreement."
    response = processor.process_query(template_query)
    print(response)
    processor.process_llm_response(response)

