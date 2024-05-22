from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import io
import numpy as np
from re import S
import torch
import librosa
from env import AttrDict
from mpnet_datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MPNet
import soundfile as sf
from transformers import pipeline
from datasets import Dataset
import gc
from torch.nn.attention import SDPBackend, sdpa_kernel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:512'
os.environ['LRU_CACHE_CAPACITY'] = '1'
os.environ['MKL_DISABLE_FAST_MM'] = '0'

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print('Complete.')
    return checkpoint_dict

class ASRManager:
    def __init__(self, denoiser_checkpoint_file):
        # initialize the model here
        torch.backends.cudnn.benchmark = False
        
        print('Initializing Denoising Model..')
        
        config_file = os.path.join(os.path.split(denoiser_checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        
        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        torch.manual_seed(self.h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.h.seed)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        
        self.model = MPNet(self.h).eval().to(self.device)
        
        state_dict = load_checkpoint(denoiser_checkpoint_file, self.device)
        self.model.load_state_dict(state_dict['generator'])
        
        print('Denoising Model Initialized.')
        
        print('Initializing Transcription Model...')
        
        self.transcription_pipe = pipeline("automatic-speech-recognition", model="distilwhisper_finetune", generate_kwargs={"language": "en"}, device=self.device)
        
        print('Transcription Model Initialized.')
    
    def process_example(self, example):
        noisy_wav, samplerate = sf.read(io.BytesIO(example['audio_bytes']))
        
        # 16 kHz sample rate
        assert samplerate == 16000
        
        del samplerate
        
        noisy_wav = torch.tensor(noisy_wav, device=self.device, dtype=torch.float)
        norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
        noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
        noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        
        example['noisy_amp'] = noisy_amp
        example['noisy_pha'] = noisy_pha
        example['norm_factor'] = norm_factor
        
        del noisy_wav, noisy_amp, noisy_pha, norm_factor
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
        return example
    
    def denoise(self, example):
        # apply MP-SENet
        # with torch.inference_mode(), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        #     amp_g, pha_g, _ = self.model(example['noisy_amp'].half(), example['noisy_pha'].half())
        #     del _
        #     example['amp_g'] = amp_g
        #     example['pha_g'] = pha_g
        
        example['amp_g'] = example['noisy_amp']
        example['pha_g'] = example['noisy_pha']
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
        return example
    
    def postprocess_example(self, example):
        audio_g = mag_pha_istft(example['amp_g'].float(), example['pha_g'].float(), self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        audio_g = audio_g / example['norm_factor']
        
        example['denoised_audio'] = audio_g.squeeze().cpu().numpy()
        
        del audio_g
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
        return example


    def digit_to_text(self, digit):
        mapping = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        return mapping[digit]

    def postprocess(self, text):
        return_text = ''
        prev_numeric = False
        for x in text.strip():
            if x.isnumeric():
                if prev_numeric:
                    return_text += ' '
                return_text += self.digit_to_text(int(x))
                prev_numeric = True
            else:
                return_text += x
                prev_numeric = False

        return return_text
    
    def convert_example(self, example):
        noisy_wav, samplerate = sf.read(io.BytesIO(example['audio_bytes']))
        
        example['denoised_audio'] = noisy_wav
        
        return example
    
    def transcribe(self, audio_bytes_list: bytes) -> str:
        ds = Dataset.from_dict({'audio_bytes': audio_bytes_list})

        # denoise audio
        print('Denoising Audio...')
        ds = ds.map(self.convert_example, remove_columns=ds.column_names).with_format('np')
        # ds = ds.map(self.process_example, remove_columns=ds.column_names).with_format('torch', device=self.device)
        # ds = ds.map(self.denoise, remove_columns=['noisy_amp', 'noisy_pha']).with_format('torch', device=self.device)
        # ds = ds.map(self.postprocess_example, remove_columns=ds.column_names).with_format('np')
        print('Audio Denoised.')
        
        transcriptions = []
        
        # perform ASR transcription
        print('Transcribing Audio...')
        for transcribed_text in self.transcription_pipe(list(ds['denoised_audio'])):
            transcriptions.append(self.postprocess(transcribed_text['text']))
        print('Audio Transcribed.')
        
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()
        #     gc.collect()
        
        return transcriptions
