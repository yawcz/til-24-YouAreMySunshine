from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
from transformers import pipeline


app = FastAPI()

checkpoint = "distilbert-finetuned-ner/checkpoint-858"

token_classifier = pipeline(
    "token-classification", model=checkpoint, aggregation_strategy="simple"
)

def find_entity_index(entity_list, target_group):
    return [index for index, entity in enumerate(entity_list) if entity['entity_group'] == target_group]

str2num = {
    "zero" : "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9",
}

@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/extract")
async def extract(instance: Request):
    """
    Performs QA extraction given a context string

    returns a dictionary with fields:

    {
        "heading": str,
        "target": str,
        "tool": str,
    }
    """
    # get transcription, and pass to NLP model
    request_dict = await instance.json()

    predictions = []
    for instance in request_dict["instances"]:
        # each is a dict with one key "transcript" and the transcription as a string
        res = token_classifier(instance["transcript"])
        
        # extract heading
        heading = res[find_entity_index(res, "HEAD")[0]]["word"] # "one six nine"
        heading = heading.lower().split() 
        heading_num = [str2num[i] for i in heading]
        heading = ''.join(heading_num)
        
        # extract target
        target = res[find_entity_index(res, "HEAD")[0]]["word"]
        
        # extract tool
        tool = res[find_entity_index(res, "HEAD")[0]]["word"]
        
        answers = {"heading": heading, "target": target, "tool": tool}
        predictions.append(answers)

    return {"predictions": predictions}
