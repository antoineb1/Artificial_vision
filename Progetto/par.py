from transformers import ViltProcessor, ViltForQuestionAnswering
import requests, os, time, cv2
from PIL import Image

class ViLTPAR:

    def __init__(self, vilt_model):
        self.processor = ViltProcessor.from_pretrained(vilt_model)
        self.model = ViltForQuestionAnswering.from_pretrained(vilt_model)

        self.gender_question = "What is the gender of the person?"
        self.hat_question = "Is the person wearing a hat?"
        self.bag_question = "Is the person carrying a bag or a backpack?"
        self.upper_clothing_question = "What color is the upper clothing?"
        self.lower_clothing_question = "What color is the lower clothing?"

        return

    def extract_attributes(self, image):

        answers = []
        feature_list = []

        try:
            questions = [
                self.processor(image, self.gender_question, return_tensors='pt'), # gender
                self.processor(image, self.hat_question, return_tensors='pt'), # hat
                self.processor(image, self.bag_question, return_tensors='pt'), # bag
                self.processor(image, self.upper_clothing_question, return_tensors='pt'), # upper color
                self.processor(image, self.lower_clothing_question, return_tensors='pt') # lower color
            ]
            
            answers.append(questions)

            for answer in answers:
                for i, a in enumerate(answer):
                    outputs = self.model(**a)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()

                    if i == 1 or i == 2:
                        if str(self.model.config.id2label[idx]).lower() == 'yes':
                            self.model.config.id2label[idx] = True
                        else:
                            self.model.config.id2label[idx] = False

                    feature_list.append(self.model.config.id2label[idx])

        except Exception as e:
            print("Error occured --- ", e)

        return feature_list
        
