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
load_dotenv()

class LLM:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        self.store = {}
        self.config = {"configurable": {"session_id": "root"}}
        self.model = ChatOpenAI(model="gpt-4o")
        self.with_message_history = RunnableWithMessageHistory(self.model, self.get_session_history)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def generate_response(self, text):

        response = self.with_message_history.invoke(
            [HumanMessage(content=text), SystemMessage(content="Translate the following from English into Italian")],
            config=self.config,
        )

        return response

    def generate_stream_response(self, text):
        memories = self.query_long_term_memory_db(text)
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
        #put audio stream here

    def short_term_memory_model_in_memory(self, docs, words_to_retrieve):
        vectorstore = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),)

        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
        res = retriever.batch(words_to_retrieve)
        return res
    
    def add_to_long_term_memory_db(self, docs):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings = OpenAIEmbeddings()

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        
        vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
        
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
        #vectorstore.drop_tables()
        memory = 'Memories from your vector database storing past conversations: '
        queries = vectorstore.similarity_search(words_to_retrieve, k=3)
        for query in queries:
            memory += (query.page_content + ' | ')

        return memory
    
    def delete_long_term_memory_db(self, words_to_retrieve):
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


# add memories to postgres db, run once commented code
'''llm = LLM()
docs = [
    Document(
        page_content="I met my friend Benji during on my first day at CMU.",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="I first saw my friend Julius in 15-122 at CMU in a class taught by Iliano Cervesato.",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="Professor Iliano's favorite pasta was Spaghetti Alle Invariante.",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center in Budapest",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]
llm.add_to_long_term_memory_db(docs)'''

# generate response that involves memory call from db once the code above in comments was run once
llm = LLM()
llm.generate_stream_response("Hi my name is Rob.")
llm.generate_stream_response("Recall my name.")
llm.generate_stream_response("Tell me all my friends that I met at CMU.")