# --------------------------------------------------------------------------------
# File : agents.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : This file contains the models for the AI Agent application.
# Purp : This contains the code for the LLMs (Large Language Models) that
#        will be used by the AI Agent application.
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Import the required libraries
# gemini-1.5-pro
# gemini-1.5-flash
# gemini-1.0-pro
# igpt4_turbo
# --------------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv() 

import os
import re
import json
import asyncio
import requests
from datetime import datetime

from anthropic import Anthropic
from anthropic import BadRequestError

from tavily import TavilyClient

from openai import OpenAI

import google.generativeai as genai

# --------------------------------------------------------------------------------
# Instantiate the clients for the various LLMs
# --------------------------------------------------------------------------------

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --------------------------------------------------------------------------------
# Safety settings for ggl
# --------------------------------------------------------------------------------


ggl_safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

# --------------------------------------------------------------------------------
# Client for igpt since it doesn't exist
# TODO: cache the token
# --------------------------------------------------------------------------------

IGPT_KEY = os.environ['IGPT_KEY']
IGPT_SECRET = os.environ['IGPT_SECRET']
IGPT_AUTH_URI = os.environ['IGPT_AUTH_URI']
IGPT_INF_URI = os.environ['IGPT_INF_URI']

class iGPT:

    def __init__(self, key: str, secret: str,
                 model: str = 'gpt-4-turbo',
                 temperature: float = 0.95,
                 top_p: float = 0.85,
                 frequency_penalty: float = 0,
                 presence_penalty: float = 0,
                 max_tokens: int = 4096):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {'grant_type': 'client_credentials',
                'client_id': key, 'client_secret': secret}
        response = requests.post(IGPT_AUTH_URI, headers=headers, data=data)
        if response.status_code == 200:
            self._token = json.loads(response.content)['access_token']
        else:
            raise ValueError(f"Can't get the auth token  {response.status_code}: {response.text}")

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens


    def generate(self, conversation: list[dict], correlationId: str = "iGPT design agents", ):
        '''
        [
            {
            "role": "system",
            "content": "Summarize everything to as few words as possible."
            },
            {
            "role": "user",
            "content": "Tell me a story about Little Red Riding Hood"
            }
        ]
        '''
        prompt = {
            "correlationId": correlationId,
            "options": {
                "temperature": self.temperature,
                "top_P": self.top_p,
                "frequency_Penalty": self.frequency_penalty,
                "presence_Penalty": self.presence_penalty,
                "max_Tokens": self.max_tokens,
                "model": self.model,
            },
            "conversation": conversation
        }

        headers = {"Authorization": f"Bearer {self._token}",
                   "Content-Type": "application/json"} 
        response = requests.post(IGPT_INF_URI, headers=headers, data=json.dumps(prompt))
        if response.status_code == 200:
            return json.loads(response.content)
        else:
            return f"iGPT Generate Error  {response.status_code}: {response.text}"

# --------------------------------------------------------------------------------
# Done :)
# --------------------------------------------------------------------------------
