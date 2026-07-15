import os
import json
import logging

METADATA_PATH = "data/dataset/metadata.json"

class Metadata:
    def __load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except json.JSONDecodeError:
                logging.warning("Файл метадаты поврежден, будет создан новый.")
                self.metadata = {}

    def __init__(self, file_path=METADATA_PATH):
        self.metadata = {}
        self.file_path = file_path
        self.__load()

    def add(self, filepath: str, item: dict):
        """Добавляет новую запись"""
        filename = os.path.basename(filepath)
        self.metadata[filename] = {
            "title": item["title"],
            "tags": item["tags"],
            "raw_path": filepath
        }

    def update(self, filename: str, updates: dict):
        """Обновляет существующую запись"""
        if filename in self.metadata:
            self.metadata[filename].update(updates)
        else:
            logging.warning(f"Попытка обновить несуществующий файл: {filename}")

    def save(self):
        """Сохраняет метадату на диск"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
