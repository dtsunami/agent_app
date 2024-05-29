#---------------------------------------------------------------------------------
# File : agents.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : This file contains the models for the AI Agent application.
# Purp : This contains the code for the LLMs (Large Language Models) that
#        will be used by the AI Agent application.
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Import the required libraries
# gemini-1.5-pro
# gemini-1.5-flash
# gemini-1.0-pro
#---------------------------------------------------------------------------------


import os
import re
import json
import asyncio
from datetime import datetime

from anthropic import Anthropic
from anthropic import BadRequestError

from tavily import TavilyClient

from openai import OpenAI

import google.generativeai as genai

#---------------------------------------------------------------------------------
# Instantiate the clients for the various LLMs
#---------------------------------------------------------------------------------

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

#---------------------------------------------------------------------------------
# Run the appropriate model for task instructions
#---------------------------------------------------------------------------------


async def run_model(model_name: str, instructions: str, previous_tasks: dict[str, str] = None):
    # Placeholder for running the model, can be orchestrator, refiner or subagent
    await asyncio.sleep(1)  # Simulating processing time
    return f"({model_name}) output for goal: {goal}"


#----------------------------------------------------------------------(and never were)
#---------------------------------------------------------------------------------
