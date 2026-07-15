from urllib.parse import urlparse
from typing import Optional, Any
import requests
import logging
import json
import io
import os
from metadata import Metadata

RAW_TXT_PATH = "data/raw/references.txt"
RAW_AUDIO_DIR = "data/raw/audio"

class ReferencesIterator:
    def __init__(self, file: io.TextIOWrapper):
        self.file = file
    
    def __read_line(self) -> Optional[str]:
        line = self.file.readline()
        if line:
            return line
        return None
    
    def __read_tag(self, tag) -> Optional[str]:
        line = self.__read_line()
        if line is None:
            return None
        return line[len(f'{tag}: '):].strip()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        title = self.__read_tag('title')
        url   = self.__read_tag('url')
        tags  = self.__read_tag('tags')
        self.__read_line()

        if title is None:
            raise StopIteration

        tags = tags.split(', ')
        return {
            'title': title,
            'url': url,
            'tags': tags
        }

def download_file(url: str, dest_dir: str, title: str) -> tuple[Optional[str], bool]:
    logging.info(f"Скачивание: {title}...")

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    dest_path = os.path.join(dest_dir, filename)
    
    if os.path.exists(dest_path):
        logging.info(f"Файл уже существует: {filename}")
        return dest_path, False
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"Успешно скачано: {filename}")
        return dest_path, True
    except Exception as e:
        logging.error(f"Ошибка при скачивании {url}: {e}")
        return None, False

def main():
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    with open(RAW_TXT_PATH, mode='r', encoding='utf-8') as f:
        metadata = Metadata(METADATA_PATH)

        try:
            for item in ReferencesIterator(f):
                url = item["url"]
                filepath, downloaded = download_file(url, RAW_AUDIO_DIR, item["title"])
                if filepath:
                    metadata.add(filepath, item)
        finally:
            metadata.save()

if __name__ == "__main__":
    main()
