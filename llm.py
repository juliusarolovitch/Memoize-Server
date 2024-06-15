import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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
            [HumanMessage(content=text)],
            config=self.config,
        )

        return response

    def generate_stream_response(self, text):
        for r in self.with_message_history.stream(
            {
                "input": [HumanMessage(content=text)],
                "language": "English",
            },
            config=self.config, ):
            print(r.content, end = "")
        print("\n")
        #put audio stream here

    def short_term_memory_model(self, docs, words_to_retrieve):
        vectorstore = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),)

        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
        res = retriever.batch(words_to_retrieve)
        return res
    
    def long_term_memory_model(self, docs, words_to_retrieve):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        collection_name = "my_docs"
        embeddings = CohereEmbeddings()

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        vectorstore.drop_tables()
        vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
        res = vectorstore.similarity_search(words_to_retrieve, k=10)

        return res
            
llm = LLM()
llm.generate_stream_response("Hi my name is Rob.")
llm.generate_stream_response("Recall my name.")
llm.generate_stream_response("Tell us the best string stream to audio stream library in Python available for my friend Julius to read output from langchain stream.")

'''docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
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
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]'''