from fastapi import FastAPI, Request
import base64
from ASRManager import ASRManager

from datasets import Dataset
import numpy as np

app = FastAPI()

asr_manager = ASRManager(denoiser_checkpoint_file='best_ckpt/g_best')


@app.get("/health")
def health():
    return {"message": "health ok"}

def denoise(audio_bytes):
    return np.array([1, 2, 3])

def process_example(example):
    example['audio'] = denoise(example['audio'])
    return example

@app.post("/stt")
async def stt(request: Request):
    """
    Performs ASR given the filepath of an audio file
    Returns transcription of the audio
    """

    # get base64 encoded string of audio, convert back into bytes
    input_json = await request.json()

    audio_bytes_list = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        audio_bytes_list.append(base64.b64decode(instance["b64"]))
        
    predictions = asr_manager.transcribe(audio_bytes_list)
    
    return {"predictions": predictions}
