# memory.py
import os
import pathlib
import time
import ujson

MEMORY_FOLDER = "./memory/"

class DialogEntry:
    def __init__(self, instruction, response):
        self.instruction = instruction
        self.response = response

def save_conversation_memory(memory):
    """
    Saves the entire conversation as a JSON file in ./memory/ with a unique timestamp.
    """
    memory_path = pathlib.Path(MEMORY_FOLDER)
    if not memory_path.exists():
        memory_path.mkdir(parents=True)

    timestamp = str(int(time.time()))
    filename = f"memory_{timestamp}.json"
    file_path = memory_path.joinpath(filename)

    # Format the conversation memory as a list of dictionaries
    formatted_memory = [{"instruction": dialog[0], "response": dialog[1]} for dialog in memory]

    with open(file_path, "w", encoding="utf-8") as file:
        ujson.dump(formatted_memory, file, ensure_ascii=False, indent=2)

    return file_path

def export_conversation(memory, export_file="codephreak_chat_history.json"):
    """
    Exports the current conversation memory as a downloadable JSON file.
    """
    formatted_memory = [{"instruction": dialog[0], "response": dialog[1]} for dialog in memory]
    with open(export_file, "w", encoding="utf-8") as file:
        ujson.dump(formatted_memory, file, ensure_ascii=False, indent=2)
    return export_file

def load_conversation_memory(file_path):
    """
    Loads a conversation memory file and returns it as a list of dictionaries.
    """
    if not pathlib.Path(file_path).exists():
        return []
    with open(file_path, "r", encoding="utf-8") as file:
        return ujson.load(file)
