import os
import json
import torch
import torchaudio
import torchaudio.transforms as T

PROCESSED_DIR = "data/processed"
METADATA_PATH = "data/dataset/metadata.json"
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

    if not os.path.exists(METADATA_PATH):
        print("Сначала запустите скачивание файлов (download.py)")
        return
        
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    for filename, info in metadata.items():
        raw_path = info["raw_path"]
        if not os.path.exists(raw_path):
            continue
            
        print(f"Предобработка: {filename}")
        processed_waveform = preprocess_audio(raw_path)
        
        processed_filename = os.path.splitext(filename)[0] + ".wav"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        
        torchaudio.save(processed_path, processed_waveform, TARGET_SAMPLE_RATE)
        
        metadata[filename]["processed_path"] = processed_path

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()