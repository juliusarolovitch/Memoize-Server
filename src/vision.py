from openai import OpenAI
import base64
import requests
import cv2

class Images:
    def __init__(self, urls, text, local_images, apikey):
      self.urls = urls #list of urls
      self.text = text #string prompt
      self.apikey = apikey #string your api key
      self.local_images = local_images #list of image paths on local drive
      self.client = OpenAI(api_key = apikey)

    def url_images_prompt(self):
      content = [{"type": "text", "text":self.text}]
      for url in self.urls:
        prompt_dict = {
              "type": "image_url",
              "image_url":{"url":url}
          }
        content.append(prompt_dict)
      response = self.client.chat.completions.create(
         model = "gpt-4o",
         messages = [{
            "role":"user",
            "content":content
         }],
         max_tokens = 2000
         )
      
      return response.choices[0].message.content
    
    def encode_image(self, image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
    def local_images_prompt(self):
      headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.apikey}"
      }

      content = [{"type": "text", "text":self.text}]

      for path in self.local_images:
        base64_image = self.encode_image(path)
        prompt = {
           "type": "image_url",
           "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
           }
        }
        content.append(prompt)

      messages = [{"role":"user", "content":content}]
      payload = {"model": "gpt-4o", "messages": messages, "max_tokens":2000}
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      
      return response.json()["choices"][0]["message"]["content"]

class Video:
  def __init__(self, file, text, apikey):
    self.file = file #complete path here
    self.text = text
    self.client = OpenAI(api_key=apikey)

  def read_frames(self):
    video = cv2.VideoCapture(self.file)
    base64Frames = []
    while video.isOpened():
      success, frame = video.read()
      if not success:
        break
      _, buffer = cv2.imencode(".jpg", frame)
      base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return base64Frames
  
  def video_prompt(self, frame_rate=50):
    base64Frames = self.read_frames()

    PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            self.text,
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::frame_rate]),
        ],
    },
    ]

    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 2000,
    }

    result = self.client.chat.completions.create(**params)
    return result.choices[0].message.content



# SOME ADDITIONAL REF NOTES FOR ONLY IMAGE AND ONLY VIDEO INPUT
# For reference: images, url
'''
client = OpenAI()
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What are in these images? Is there any difference between them?",
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)
print(response.choices[0]) '''

# For reference: images, local drive
'''
import base64
import requests

# OpenAI API Key
api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json()) '''

# For reference: video input

'''client = OpenAI() #provide api_key = "YOUR API KEY" in parentheses
video = cv2.VideoCapture("bison.mp4") #provide full path here
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)'''