import os
from llm import LLM
from vision import Images, Video
from dotenv import load_dotenv
load_dotenv()

class DataPipeline:
    def __init__(self, folder : str):
        self.folder = folder
        self.reader = LLM()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    def run(self):
        for filename in os.listdir(self.folder):
            file_path = os.path.join(self.folder, filename)
            name, extension = os.path.splitext(filename)

            if extension == '.txt':
                self.reader.upload_personal_info_structured_rag(file_path)

            elif extension == '.mp3':
                prompt = 'Describe the image in 3 sentences.'
                text = Images(None, prompt, file_path, self.openai_api_key).local_images_prompt()
                self.reader.add_to_long_term_memory_db(text)

            elif extension == '.mp4':
                prompt = 'Describe the video in up to 10 sentences.'
                text = Video(file_path, prompt, self.openai_api_key).video_prompt()
                self.reader.add_to_long_term_memory_db(text)

        return 
    
model = LLM()
model.delete_long_term_memory_db()
DataPipeline('/Users/deven/SynologyDrive/OnlyBrains/Memoize-Server/server/src/datapipeline_tests').run()

while True:
    conversation = ""
    prompt = input("User: Rob. Text: ")
    response = model.generate_stream_response(prompt, memory_storage='disk')
    conversation = prompt + "\n" + response
    model.add_to_long_term_memory_db(conversation)
    conversation = ""
