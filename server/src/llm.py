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
import time
from openai import AsyncOpenAI
import shutil
import subprocess
import asyncio
import websockets
import json
import base64
load_dotenv()
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVEN_API_KEY')
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
        self.aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

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

        return response.content

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
    
    def is_installed(self, lib_name):
        return shutil.which(lib_name) is not None

    async def text_chunker(self, chunks):
        """Split text into chunks, ensuring to not break sentences."""
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        buffer = ""

        async for text in chunks:
            if text is None:
                continue

            if buffer.endswith(splitters):
                yield buffer + " "
                buffer = text
            elif text.startswith(splitters):
                yield buffer + text[0] + " "
                buffer = text[1:]
            else:
                buffer += text

        if buffer:
            yield buffer + " "

    async def stream(self, audio_stream, start_time):
        """Stream audio data using ffplay player."""
        if not self.is_installed("ffplay"):
            raise ValueError(
                "ffplay not found, necessary to stream audio. "
                "Install instructions: https://ffmpeg.org/download.html"
            )

        ffplay_process = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        first_audio_chunk = True
        async for chunk in audio_stream:
            if chunk:
                if first_audio_chunk:
                    first_audio_chunk = False
                    elapsed_time = time.time() - start_time
                    print(f"Time to first audio: {elapsed_time:.2f} seconds")
                ffplay_process.stdin.write(chunk)
                ffplay_process.stdin.flush()

        if ffplay_process.stdin:
            ffplay_process.stdin.close()
        ffplay_process.wait()

    async def text_to_speech_input_streaming(self, voice_id, text_iterator, start_time):
        """Send text to ElevenLabs API and stream the returned audio."""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2"

        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": ELEVENLABS_API_KEY,
            }))

            async def listen():
                """Listen to the websocket for audio data and stream it."""
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("audio"):
                            yield base64.b64decode(data["audio"])
                        elif data.get('isFinal'):
                            break
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed")
                        break

            listen_task = asyncio.create_task(self.stream(listen(), start_time))

            async for text in self.text_chunker(text_iterator):
                await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

            await websocket.send(json.dumps({"text": ""}))

            await listen_task

    async def chat_completion(self, text, memory_storage="db"):
        """Retrieve text from OpenAI and pass it to the text-to-speech function."""
        if memory_storage == "db":
            memories = self.query_long_term_memory_db(text)
        elif memory_storage == 'disk':
            memories = self.query_long_term_memory_disk(text)
        else:
            memories = ""

        message = memories + '| Prompt:' + text + "| Role: Speak in first person view as if you were the user. You are the digital twin of the user. You think the same way, you know the same things and hold the same opinions as the user. Never admit that you are ChatGPT in any circumstance."
        response = await self.aclient.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': message}],
        temperature=1, stream=True)

        async def text_iterator():
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield delta.content

        start_time = time.time()  
        await self.text_to_speech_input_streaming(VOICE_ID, text_iterator(), start_time)
    
# #####DEMO#######
# 1. start docker: docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
# 2. Only do this step if you are running the code for the first time (rag):
#   a. llm = LLM()
#   b. rag_txt_file = 'rag.txt'
#   c. personal_info = llm.upload_personal_info_structured_rag(rag_txt_file, to = 'disk')
# 3. run code below: (note that step only has to be run once, because after then everything is stored in db on docker)
if __name__ == "__main__":
    user_query = "Tell me the history of ur favorite country in 30 sentences after introducing yourself."
    llm = LLM()
    asyncio.run(llm.chat_completion(user_query))


#JUST SOME PAST NOTES
# ###Initialize LLM
# llm = LLM()

# ###Clean vector DB
# llm.delete_long_term_memory_disk()

# ###RAG for structured data (example user: Bob) MAY TAKE A WHILE.
# rag_txt_file = 'rag.txt'
# personal_info = llm.upload_personal_info_structured_rag(rag_txt_file, to = 'disk')

# ###RAG for unstructured conversations
# # llm.upload_conversations_unstructered_rag(conversations)

# ###FINETUNE for voice and speech style

# ###Simulate conversation
# while True:
#     conversation = ""
#     prompt = input("User: Rob. Text: ")
#     response = llm.generate_stream_response(prompt, memory_storage='disk')
#     conversation = prompt + "\n" + response
#     llm.add_to_long_term_memory_disk(conversation)
#     conversation = ""