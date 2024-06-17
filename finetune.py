from openai import OpenAI
import time

class FineTune:
    
    def __init__(self, training_file, apikey, model="gpt-3.5-turbo-1106"):
      self.training_file = training_file
      self.model = model
      self.apikey=apikey
      self.client = OpenAI(api_key=apikey)
    def train(self):
      data = self.client.files.create(
      file=open(self.training_file, "rb"),
      purpose="fine-tune")

      res = self.client.fine_tuning.jobs.create(
      training_file=data.id, 
      model=self.model)
      
      retrieved = self.client.fine_tuning.jobs.retrieve(res.id)
      print("LLM model fine tuning in progress...")
      while retrieved.fine_tuned_model == None: 
         retrieved = self.client.fine_tuning.jobs.retrieve(res.id)
         time.sleep(10)
      
      return retrieved.fine_tuned_model

