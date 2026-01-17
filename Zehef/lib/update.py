import os
import json
from .Requests import Request
from .colors import *

class Version_Checker:
    async def checker():
        # Get the absolute path to config.json
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

        # Open and read the config file
        with open(CONFIG_PATH, "+r", encoding='utf-8') as file:
            reader = json.loads(file.read())  # ✅ Correct indentation

        version = reader['version']['number']
        name = reader['version']['name']
        
        # Fetch the latest version from GitHub
        r = await Request("https://raw.githubusercontent.com/N0rz3/Zehef/master/config.json").get()
        conf = json.loads(r.text)
        current_version = conf['version']['number']

        if version == current_version:
            pass
        else: 
            pass
