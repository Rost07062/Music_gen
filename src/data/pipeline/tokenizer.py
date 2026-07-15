import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import EncodecModel, AutoProcessor
from metadata import Metadata 

TOKENS_DIR = "data/tokens"
TARGET_SAMPLE_RATE = 48000 # EnCodec 48kHz
TARGET_TIME_STEPS = 128
TARGET_CODEBOOKS = 32
MODEL_NAME = "facebook/encodec_48khz" 

def load_model_and_processor():
    print(f"Загрузка модели {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = EncodecModel.from_pretrained(MODEL_NAME)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Модель загружена на {device.upper()}")
    return model, processor, device

def tokenize_audio(file_path, model, processor, device):
    waveform, sr = torchaudio.load(file_path)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1) 
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    
    if sr != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        
    if waveform.dim() > 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = processor(waveform, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE)
    input_values = inputs["input_values"].to(device)
    
    with torch.no_grad():
        encoded = model.encode(input_values, bandwidth=24.0)
        
    audio_codes = encoded.audio_codes         # (C, B, D, N)
    codes = audio_codes.squeeze(1)            # (C, D, N)
    codes = codes.permute(2, 0, 1)            # (N, C, D)
    codes = codes.reshape(codes.shape[0], -1) # (N, C * D)
    
    current_time_steps = codes.shape[0]
    if current_time_steps > TARGET_TIME_STEPS:
        codes = codes[:TARGET_TIME_STEPS, :]
    elif current_time_steps < TARGET_TIME_STEPS:
        pad_size = TARGET_TIME_STEPS - current_time_steps
        padding = torch.zeros((pad_size, TARGET_CODEBOOKS), dtype=torch.long, device=device)
        codes = torch.cat([codes, padding], dim=0)
        
    return codes.cpu()

def main():
    os.makedirs(TOKENS_DIR, exist_ok=True)
    
    metadata = Metadata()
    
    if not metadata.metadata:
        print("Метаданные не заполнены. Нужно запустить download.py и preprocess.py")
        return

    model, processor, device = load_model_and_processor()

    for filename, info in metadata.metadata.items():
        processed_path = info.get("processed_path")
        if not processed_path or not os.path.exists(processed_path):
            print(f"Пропуск {filename}: нет предобработанного файла")
            continue

        print(f"Токенизация: {filename}...")
        
        tokens = tokenize_audio(processed_path, model, processor, device)
        
        token_filename = os.path.splitext(filename)[0] + ".pt"
        token_path = os.path.join(TOKENS_DIR, token_filename)
        torch.save(tokens, token_path)
        
        metadata.update(filename, {
            "tokens_path": token_path,
            "token_shape": list(tokens.shape)
        })

    metadata.save()
    print(f" Токены сохранены в {TOKENS_DIR}")

if __name__ == "__main__":
    main()
