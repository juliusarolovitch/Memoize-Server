import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_cohere import CohereEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
load_dotenv()

#LOGIC: there are two stages: 
# Pre-training / RAG: the document with questions and answers is stored in the vector db
# Long-term memory and responses: whenever the LLM is called, it provides a reponse and the conversation is saved

class LLM:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        self.store = {}
        self.config = {"configurable": {"session_id": "root"}}
        self.model = ChatOpenAI(model="gpt-4o")
        self.with_message_history = RunnableWithMessageHistory(self.model, self.get_session_history)

    def upload_personal_info_structured_rag(self, file_path, to = 'db'):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                memory = ""
                for line in lines:
                    if line == '\n':
                        if to == 'db':
                            self.add_to_long_term_memory_db(memory)
                        elif to == 'disk':
                            self.add_to_long_term_memory_disk(memory)
                        else:
                            pass
                        memory = ""
                    else:
                        memory += line
        except FileNotFoundError:
            print(f"The file at {file_path} was not found.")

        return

    def upload_conversations_unstructered_rag_db(self, conversations):

        doc = [Document(page_content=conversations)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(doc)
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings = OpenAIEmbeddings()
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        vectorstore.add_documents(docs)

    def upload_conversations_unstructered_rag_disk(self, conversations):

        doc = [Document(page_content=conversations)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(doc)
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings = OpenAIEmbeddings()
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        vectorstore.add_documents(docs)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def generate_response(self, text):

        response = self.with_message_history.invoke(
            [HumanMessage(content=text), SystemMessage(content=text)],
            config=self.config,
        )

        return response

    def generate_stream_response(self, text, memory_storage="db"):

        if memory_storage == "db":
            memories = self.query_long_term_memory_db(text)
        elif memory_storage == 'disk':
            memories = self.query_long_term_memory_disk(text)
        else:
            memories = ""

        message = memories + '| Prompt:' + text + "| Role: Speak in first person view as if you were the user. You are the digital twin of the user. You think the same way, you know the same things and hold the same opinions as the user. Never admit that you are ChatGPT in any circumstance."
        for r in self.with_message_history.stream(
            {
                "input": [HumanMessage(content=message), 
                          SystemMessage(content="You are a chatbot that acts as a digital twin of the user. Everything you learn about the user you remember and store it. When asked, you answer questions like the user who is your real life twin (same style, tone, logic)")],
                "language": "English"
            },
            config=self.config):
            print(r.content, end = "")

        print("\n")
        return message
        
        #put audio stream here

    def add_to_long_term_memory_disk(self, text):
        docs = [Document(page_content=text,
                         metadata={"speakers": "", "location": "", "topic": ""})]
        embeddings = OpenAIEmbeddings()

        # load it into Chroma
        db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        
        return
    
    def delete_long_term_memory_disk(self):
        embeddings = OpenAIEmbeddings()
        docs = [Document(page_content="",
                         metadata={"speakers": "", "location": "", "topic": ""})]
        db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        db.delete_collection()
        return
    
    def query_long_term_memory_disk(self, words_to_retrieve):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        memory = 'Memories from your vector database storing past conversations: '
        queries = vectorstore.similarity_search(words_to_retrieve, k=20)
        for query in queries:
            memory += (query.page_content + ' | ')

        return memory
        
    def add_to_long_term_memory_db(self, text):

        docs = [Document(page_content=text,
                         metadata={"speakers": "", "location": "", "topic": ""})]

        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings = OpenAIEmbeddings()

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        
        vectorstore.add_documents(docs)
        
        return
        
    def query_long_term_memory_db(self, words_to_retrieve):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings = OpenAIEmbeddings()

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        
        memory = 'Memories from your vector database storing past conversations: '
        queries = vectorstore.similarity_search(words_to_retrieve, k=20)
        for query in queries:
            memory += (query.page_content + ' | ')

        return memory
    
    def delete_long_term_memory_db(self):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings = OpenAIEmbeddings()

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        vectorstore.drop_tables()
        
        return
    
#####DEMO#######

###Initialize LLM
llm = LLM()

###Clean vector DB
llm.delete_long_term_memory_disk()

###RAG for structured data (example user: Bob) MAY TAKE A WHILE.
rag_txt_file = 'rag.txt'
personal_info = llm.upload_personal_info_structured_rag(rag_txt_file, to = 'disk')

###RAG for unstructured conversations
# llm.upload_conversations_unstructered_rag(conversations)

###FINETUNE for voice and speech style

###Simulate conversation
while True:
    conversation = ""
    prompt = input("User: Rob. Text: ")
    response = llm.generate_stream_response(prompt, memory_storage='disk')
    conversation = prompt + "\n" + response
    llm.add_to_long_term_memory_disk(conversation)
    conversation = ""