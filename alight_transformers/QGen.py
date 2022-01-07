#!/usr/bin/env python
# coding: utf-8


# Imports:
import pdfminer
from pdfminer.high_level import extract_text
from Questgen import main
from pprint import pprint
import nltk
nltk.download('stopwords')
import numpy as np


import csv
import pandas as pd

def pdf_to_clean_text(path, first_page_num, end_page_num ):
    page_range = list(range(first_page_num-1, end_page_num))
# The most simple way to extract text from a PDF is to use extract_text:
    text = extract_text(path, page_numbers = page_range)
    text = text.split('\n\n')
    text = [x.replace('\n', '') for x in text]
    text_split = [x.split('.', 1) for x in text]
    
    # Used to convert list of list into a single list
    def flatten(t):
        return [item for sublist in t for item in sublist]

    text = flatten(text_split)

    return text

class QuestionGenerator:
    
    def __init__(self, text):
        self.text = text
        self.output = self.generate_FAQ()
        self.output_bool = self.generate_Bool()
        self.questions, self.answers, self.contexts = self.extract()
        self.answers_AP = self.generate_AP()
  
    ##########################################
    # Need to factor pdf_to_clean_text() out # Different data formats require different methods.
    ##########################################
#     def pdf_to_clean_text(self):
#     # The most simple way to extract text from a PDF is to use extract_text:
#         text = extract_text(self.path, page_numbers = self.page_range)
#         text = text.split('\n\n')
#         text = [x.replace('\n', '') for x in text]
#         text_split = [x.split('.', 1) for x in text]
#         def flatten(t):
#             return [item for sublist in t for item in sublist]

#         text = flatten(text_split)
        
#         return text
    
    
    def generate_FAQ(self):
        qg = main.QGen()

        output = []
        # For loop goes through each paragraph to generate the FAQs
        for n in range(len(self.text)):
            payload = {'input_text' : self.text[n]}
            result = qg.predict_shortq(payload)
            output.append(result)
        return output
    
    def generate_Bool(self):
        qe = main.BoolQGen()

        output_bool = []
        # For loop goes through each paragraph to generate the FAQs
        for n in range(len(self.text)):
            payload = {'input_text' : self.text[n]}
            result = qe.predict_boolq(payload)
            output_bool.append(result)
        return output_bool
    
    def extract(self):
        #Repeating the above code for all 3: questions, answers, and context
        questions=[]
        answers = []
        contexts = []

        for n in self.output:
            if 'questions' in n.keys(): #since there are empty dicts in the output liDpedicst

                for qa in n['questions']:
                    questions.append(qa['Question'])
                    answers.append(qa['Answer'])
                    contexts.append(qa['context'])
                    
        for n in self.output_bool:
            if 'Boolean Questions' in n.keys(): #since there are empty dicts in the output liDpedicst
                for qa in n['Boolean Questions']:
                    questions.append(qa) # Appending to the `questions` list from the QGen FAQ
                    contexts.append(n['Text']) # Appending to the `context` list from the QGen FAQ 
                    answers.append(None)

                    ## NOTE: there is repetition in the contexts appended from BoolQGen
        return questions, answers, contexts
    
    # Also using the AnswerPredictor from Questgen which seemed to give more sensible responses
    def generate_AP(self):
        answer = main.AnswerPredictor()
        answers_AP = []
        for q, c in list(zip(self.questions, self.contexts)):
            payload = {'input_text' : c,
                       'input_question' : q}
            a = answer.predict_answer(payload)
            answers_AP.append(a)
        return answers_AP
    
    def create_df(self):
        dict = {'Questions':self.questions,'Answers_FAQ':self.answers,'Answers_AP':self.answers_AP, 'Contexts':self.contexts}
        pd.set_option('display.max_colwidth', None)
        df = pd.DataFrame(dict)
        df['Contexts'].replace('', np.nan, inplace=True)
        df.dropna(subset=['Contexts'], inplace=True)
        return df

