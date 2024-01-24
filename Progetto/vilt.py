from transformers import ViltProcessor, ViltForQuestionAnswering
import requests, os, time
from PIL import Image

class ViLTPAR:

    def __init__(self):
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        self.gender_question = "What is the gender of the person?"
        self.hat_question = "Is the person wearing a hat?"
        self.bag_question = "Is the person carrying a bag?"
        self.upper_clothing_question = "What color is the upper clothing?"
        self.lower_clothing_question = "What color is the lower clothing?"

        self.results = []

        return

    def get_model(self):
        return self.model

    def get_results(self):
        return self.results

    def extract_attributes(self, id, image):
        # image.save("id" + str(id + 1) + ".jpg")

        try:
            attributes = [
                ('gender', self.processor(image, self.gender_question, return_tensors='pt')),
                ('hat', self.processor(image, self.hat_question, return_tensors='pt')),
                ('bag', self.processor(image, self.bag_question, return_tensors='pt')),
                ('upper_color', self.processor(image, self.upper_clothing_question, return_tensors='pt')),
                ('lower_color', self.processor(image, self.lower_clothing_question, return_tensors='pt'))
            ]
            self.results.append(attributes)
        except Exception as e:
            print("Error occured: ", e)


    def print_attributes(self):
            for attr in self.results:
                for a in attr:
                    outputs = self.model(**a[1])
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()
                    print(a[0], self.model.config.id2label[idx], '\n')
                print('\n/////////////\n\n')
