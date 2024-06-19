import os
import json
import base64
import urllib.parse
import websockets
from websockets.sync.client import connect
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

OMIT = typing.cast(typing.Any, ...)


class LLM:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        self.store = {}
        self.config = {"configurable": {"session_id": "root"}}
        self.model = ChatOpenAI(model="gpt-4o")
        self.with_message_history = RunnableWithMessageHistory(
            self.model, self.get_session_history)
        self.eleven_labs_api_key = os.getenv('ELEVEN_LABS_API_KEY')
        self.voice_id = os.getenv('ELEVEN_LABS_VOICE_ID')

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def generate_response(self, text):
        response = self.with_message_history.invoke(
            [HumanMessage(content=text), SystemMessage(
                content="Translate the following from English into Italian")],
            config=self.config,
        )
        return response

    def text_chunker(self, chunks: typing.Iterator[str]) -> typing.Iterator[str]:
        """Used during input streaming to chunk text blocks and set last char to space"""
        splitters = (".", ",", "?", "!", ";", ":", "—",
                     "-", "(", ")", "[", "]", "}", " ")
        buffer = ""
        for text in chunks:
            if buffer.endswith(splitters):
                yield buffer if buffer.endswith(" ") else buffer + " "
                buffer = text
            elif text.startswith(splitters):
                output = buffer + text[0]
                yield output if output.endswith(" ") else output + " "
                buffer = text[1:]
            else:
                buffer += text
        if buffer != "":
            yield buffer + " "

    def stream_audio(self, text: typing.Iterator[str]):
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id=eleven_monolingual_v1"
        headers = {
            "xi-api-key": self.eleven_labs_api_key,
            "Content-Type": "application/json"
        }

        with connect(url, additional_headers=headers) as socket:
            try:
                socket.send(json.dumps(
                    {
                        "text": " ",
                        "try_trigger_generation": True,
                        "voice_settings": {
                            "stability": 0.75,
                            "similarity_boost": 0.75,
                            "style": 1.0,
                            "use_speaker_boost": True
                        },
                        "generation_config": {
                            "chunk_length_schedule": [50],
                        },
                    }
                ))
            except websockets.exceptions.ConnectionClosedError as ce:
                print(f"Connection closed: {ce.reason} (code {ce.code})")

            try:
                for text_chunk in self.text_chunker(text):
                    data = {
                        "text": text_chunk,
                        "try_trigger_generation": True
                    }
                    socket.send(json.dumps(data))
                    try:
                        data = json.loads(socket.recv())
                        if "audio" in data and data["audio"]:
                            audio_chunk = base64.b64decode(data["audio"])
                            # Play or save the audio chunk here
                            with open("output_audio.mp3", "ab") as f:
                                f.write(audio_chunk)
                    except TimeoutError:
                        pass

                socket.send(json.dumps({"text": ""}))

                while True:
                    data = json.loads(socket.recv())
                    if "audio" in data and data["audio"]:
                        audio_chunk = base64.b64decode(data["audio"])
                        with open("output_audio.mp3", "ab") as f:
                            f.write(audio_chunk)
            except websockets.exceptions.ConnectionClosed as ce:
                if "message" in data:
                    print(f"Error: {data['message']} (code {ce.code})")
                elif ce.code != 1000:
                    print(f"Connection closed: {ce.reason} (code {ce.code})")

    def generate_stream_response(self, text):
        memories = self.query_long_term_memory_db(text)
        message = memories + '| Prompt:' + text + "| Role: Speak in first person view as if you were the user. You are the digital twin of the user. You think the same way, you know the same things and hold the same opinions as the user. Never admit that you are ChatGPT in any circumstance."

        text_chunks = (r.content for r in self.with_message_history.stream(
            {
                "input": [HumanMessage(content=message),
                          SystemMessage(content="You are a chatbot that acts as a digital twin of the user. Everything you learn about the user you remember and store it. When asked, you answer questions like the user who is your real life twin (same style, tone, logic)")],
                "language": "English"
            },
            config=self.config))

        self.stream_audio(text_chunks)
        print("\n")

    def short_term_memory_model_in_memory(self, docs, words_to_retrieve):
        vectorstore = Chroma.from_documents(
            docs,
            embedding=OpenAIEmbeddings(), )

        retriever = RunnableLambda(vectorstore.similarity_search).bind(
            k=1)
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

        vectorstore.add_documents(
            docs, ids=[doc.metadata["id"] for doc in docs])

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
        # vectorstore.drop_tables()
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
