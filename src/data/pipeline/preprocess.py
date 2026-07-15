import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from metadata import Metadata

PROCESSED_DIR = "data/processed"
TARGET_SAMPLE_RATE = 16000

def preprocess_audio(file_path, target_sr=TARGET_SAMPLE_RATE):
    waveform, sr = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))
        
    return waveform

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    metadata = Metadata()
    
    if not metadata.metadata:
        print("Метадата пуста. Сначала запустите download.py")
        return

    for filename, info in metadata.metadata.items():
        raw_path = info.get("raw_path")
        if not raw_path or not os.path.exists(raw_path):
            continue
            
        print(f"Предобработка: {filename}")
        processed_waveform = preprocess_audio(raw_path)
        
        processed_filename = os.path.splitext(filename)[0] + ".wav"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        
        torchaudio.save(processed_path, processed_waveform, TARGET_SAMPLE_RATE)
        
        metadata.update(filename, {
            "processed_path": processed_path
        })

    metadata.save()

if __name__ == "__main__":
    main()