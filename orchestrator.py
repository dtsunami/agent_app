#---------------------------------------------------------------------------------
# File : orchestration.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : Prompts and strategies for AI Agent orchestration
# Purp : The base prompts and iteration strategies for the AI Agent application.
# Anthropic Models
# claude-3-opus-20240229
# claude-3-sonnet-20240229
# claude-3-haiku-20240307
#---------------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv() 

import re
import json
import asyncio

from datetime import datetime

from typing import Any
from typing import Dict
from typing import Annotated
from typing import Union

import numpy as np
import pandas as pd
import os

from pydantic import BaseModel
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import AfterValidator
from pydantic import WithJsonSchema

from rich.console import Console
from rich.panel import Panel

from agents import openai_client, anthropic_client, tavily_client, genai, ggl_safety_settings, iGPT, IGPT_KEY, IGPT_SECRET

from anthropic import RateLimitError
from requests.exceptions import HTTPError

import time

import io
import zipfile


from bson import ObjectId

igpt_client = iGPT(IGPT_KEY, IGPT_SECRET)

orch_base_prompt = '''
Assess if the Objective has been fully achieved and if not, breakdown the next subtask.
Please select the next subtask that most advances the obective and create a clear,
encouraging and comprehensive prompt for a subagent to execute that subtask.
ALWAYS CHECK CODE FOR ERRORS AND USE THE BEST PRACTICES FOR CODING TASKS AND INCLUDE FIXES FOR THE NEXT SUBTASK.",
If you have any sugestions on how code can be improved or refactored, please include them in the next subtask prompt
If the objective is not yet fully achieved, select the next most appropriate subtask
and create a clear and comprehensive prompt for a subagent to execute that subtask.
'''

refiner_base_prompt = '''
Please review the sub-task and baseline results and refine them into a cohesive final output.
Add any missing documenation or details as needed.
'''

# --------------------------------------------------------------------------------
# Agent Configuration Models
# --------------------------------------------------------------------------------


def validate_object_id(v: Any) -> ObjectId:
    if isinstance(v, ObjectId):
        return v
    if ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")


PyObjectId = Annotated[
    Union[str, ObjectId],
    AfterValidator(validate_object_id),
    PlainSerializer(lambda x: str(x), return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


class ModelConfig(BaseModel):
    orchestrator_model: str = Field(...)
    refiner_model: str = Field(...)
    subagent_model: str = Field(...)
    orchestrator_prompt: str = orch_base_prompt
    refiner_prompt: str = refiner_base_prompt
    strategy: str = Field(...)
    task_iter: int = 5
    refine_iter: int = 3
    orch_max_tokens: int = 4096
    sub_max_tokens: int = 4096
    refine_max_tokens: int = 4096


class AgentConfig(BaseModel):
    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    name: str = Field(...)
    objective: str = Field(...)
    subtask_queries: dict[int, list[str]] = {}
    subtask_results: dict[int, list[str]] = {}
    era_results: list[str] = []
    files: dict[str, str] = {}
    use_search: bool = False
    include_files: bool = False
    model: ModelConfig

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# --------------------------------------------------------------------------------
# Query the orchestrator for the next task
# async
# --------------------------------------------------------------------------------


def query_orchestrator(agent: AgentConfig, idx_ref: int, era_output: str, console: Console):
    console.print(f"\n[bold]Query orchestrator model: {agent.model.orchestrator_model}[/bold]")

    results_str = "None"
    if era_output is not None:
        results_str = f"**Baseline Results**\n{era_output}"

    results = [f"**Subtask {i} Results**\n{r}" for i, r in enumerate(agent.subtask_results[idx_ref])]
    if len(results) > 0:
        results_str += "\n".join(results)

    orch_prompt = [
        "**PROMPT**\n\n",
        "In order to fully, correctly and comprehensively complete the Objective, ",
        f"{' and using the file content ' if agent.include_files else ''}",
        f"{' without forgetting anything from the previous subtask results, ' if len(results_str) > 0 else ''}",
        agent.model.orchestrator_prompt,
        "If the previous subtask results comprehensively complete all the requirements of the objective ",
        "start your response with the phrase 'Objective Complete:'. ",
        f"\n\n**Objective:**\n{agent.objective}\n\n",
        f"\n\n**Results:**\n{results_str}\n\n\n",
        "IMPORTANT, YOUR JOB IS TO GENERATE A PROMPT FOR SUBAGENT IF THE OBJECTIVE IS NOT COMPLETE!!!!\n\n\n"
    ]
    if agent.include_files:
        orch_prompt += [f'**File content ({name}):**\n{cont}\n\n' for name, cont in agent.files.items()]

    if agent.use_search:
        # TODO: rewrite the boilerplate search query
        search_query = [
            "Please also generate a JSON object containing a single 'search_query' key, ",
            "which represents a question that, when asked online, would yield important information for solving the subtask. ",
            "The question should be specific and targeted to elicit the most relevant and helpful resources. ",
            "Format your JSON like this, with no additional text before or after:\n{'search_query': '<question>'}\n"
        ]
        orch_prompt += ["".join(search_query)]

    orch_str = "".join(orch_prompt)
    orch_response = None
    if 'claude' in agent.model.orchestrator_model:
        orch_messages = [{"role": "user", "content": [{"type": "text", "text": orch_str}]}]
        while True:
            try:
                orch_response = anthropic_client.messages.create(
                    model=agent.model.orchestrator_model,
                    max_tokens=agent.model.orch_max_tokens,
                    messages=orch_messages
                )
                console.print(f"[bold green]Orchestrator output[/bold green]")
                console.print(f"[bold green]Input Tokens {orch_response.usage.input_tokens}[/bold green]")
                console.print(f"[bold green]Output Tokens {orch_response.usage.output_tokens}[/bold green]")
                response_text = orch_response.content[0].text
                break
            except RateLimitError as e:
                console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                time.sleep(60)

    elif 'gemini' in agent.model.orchestrator_model:
        model = genai.GenerativeModel(agent.model.orchestrator_model)
        orch_response = model.generate_content("".join(orch_prompt), safety_settings=ggl_safety_settings)
        try:
            response_text = orch_response.text
        except ValueError:
            # If the response doesn't contain text, check if the prompt was blocked.
            console.print(f"\n[bold red]Value Error During response.text[/bold red]")
            console.print(f"\n[bold red]Prompt Feedback : {orch_response.prompt_feedback}[/bold red]")
            console.print(f"\n[bold red]Finish Reason : {orch_response.candidates[0].finish_reason}[/bold red]")
            console.print(f"\n[bold red]Safety Ratings : {orch_response.candidates[0].safety_ratings}[/bold red]")
            response_text = "come again?"
    

    elif 'igpt' in agent.model.orchestrator_model:
        global igpt_client
        conversation = []
        role = "You are a expert at creating prompts for AI sub-agents."
        orch_prompt.append("\n\nDO NOT INCLUDE THE PHRASE 'Objective Complete:' IN YOUR RESPONSE UNTIL THE OBJECTIVE IS FULLY COMPLETED!\n\n")
        orch_str = "".join(orch_prompt)
        console.print(f"\n[bold]iGPT content length: {len(orch_str)}[/bold]")
        console.print(f"\n[bold]iGPT total length: {len(orch_str) + len(role)}[/bold]")
        conversation.append({'role': 'system', 'content': role})
        conversation.append({'role': 'user', 'content': orch_str})

        orch_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))
        if "Token has expired" in orch_response:
            igpt_client = iGPT(key=IGPT_KEY, secret=IGPT_SECRET)
            orch_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))

        if 'usage' in orch_response:
            console.print(f"[bold green]Orchestrator output[/bold green]")
            console.print(f"[bold green]Input Tokens {orch_response['usage']['promptTokens']}[/bold green]")
            console.print(f"[bold green]Output Tokens {orch_response['usage']['completionTokens']}[/bold green]")
        try:
            response_text = orch_response['currentResponse']
        except TypeError as e:
            console.print(f"[bold red]Error querying orchestrator model {orch_response}[/bold red]")
            response_text = "come again?"

    elif 'gpt' in agent.model.orchestrator_model:
        raise NotImplementedError("GPT-4 is not yet supported")
    else:
        raise ValueError(f"Unsupported orchestrator model: {agent.model.orchestrator_model}")

    # response text
    response_pnl = Panel(response_text, 
                        title=f"[bold green]Orchestrator[/bold green]",
                        title_align="",
                        border_style="yellow",
                        subtitle="SubAgent Task")
    console.print(response_pnl)

    search_query = None
    if agent.use_search and "{'search_query': '" in response_text:
        search_query = response_text.split("{'search_query': '")[1].split("'}")[0]
        response_pnl = Panel(search_query,
                            title=f"[bold green]Search[/bold green]",
                            title_align="",
                            border_style="green",
                            subtitle="Query")
        console.print(response_pnl)

    return response_text, search_query


# --------------------------------------------------------------------------------
# Search current data for the next task
# --------------------------------------------------------------------------------


def query_search_provider(query: str, provider: str, console: Console):
    if provider == "tavily":
        try:
            search_response = tavily_client.qna_search(query=query)
        except HTTPError as e:
            try:
                search_response = tavily_client.qna_search(query=query)
            except HTTPError as e:
                search_response = "Error querying Tavily"
    else:
        raise ValueError(f"Unsupported search provider: {provider}")
    response_pnl = Panel(search_response,
                        title=f"[bold green]Search[/bold green]",
                        title_align="",
                        border_style="green",
                        subtitle="Result")
    console.print(response_pnl)
    return search_response


# --------------------------------------------------------------------------------
# Query the refiner for final output
# --------------------------------------------------------------------------------


def refine_output_continue(agent: AgentConfig, idx_ref: int, era_output: str, console: Console):
    console.print("\n[bold]Refining the Subtask results[/bold]")

    subtask_str = '\n\n'.join([f"**Subtask {i}**\n{r}" for i, r in enumerate(agent.subtask_results[idx_ref])])
    refiner_prompt = [
        f"**Objective:**\n{agent.objective}\n\n",
        ]
    if era_output is not None:
        refiner_prompt.append(f"**Baseline result:**\n{era_output}\n\n",)

    refiner_prompt += [
        f"**Results:**\n{subtask_str}\n\n",
        "**PROMPT:**\n\n",
        agent.model.refiner_prompt,
        "Provide a relevent, brief and descriptive name for the project and include it in the final output in the format <project_name>name</project_name>. ",
        "INCLUDE THE FOLLOWING:\n",
        "1. Folder Structure: Provide the folder structure as a valid JSON object, ",
        "where each key represents a folder or file, and nested keys represent subfolders. ",
        "Use null values for files. Ensure the JSON is properly formatted without any syntax errors. ",
        "Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, "
        "separating items with commas as necessary. Wrap the JSON object in <folder_structure> tags.\n",
        "2. Code, Data or Image Files: For each  file wrap the file contents in tags like this <file name='filename'>contents</file>. DO NOT INCLUDE THE triple backticks ``` and filetype!!!!!!!! ",
        ]
    refiner_str = "".join(refiner_prompt)

    if "claude" in agent.model.refiner_model:
        idx_try = 0
        while True:
            try:
                content = [{"type": "text", "text": refiner_str}]
                refiner_response = anthropic_client.messages.create(
                    model=agent.model.refiner_model,
                    max_tokens=agent.model.refine_max_tokens,
                    messages=[{"role": "user", "content": content}]
                )
                console.print(f"[bold green]Refined output, prompt length "
                              f"{len(refiner_str)}[/bold green]")
                console.print(f"[bold green]Input Tokens "
                              f"{refiner_response.usage.input_tokens}"
                              "[/bold green]")
                console.print(f"[bold green]Output Tokens "
                              f"{refiner_response.usage.output_tokens}"
                              "[/bold green]")
                refined_output = refiner_response.content[0].text

                # response text
                response_pnl = Panel(refined_output, 
                                    title=f"[bold magenta]Refiner Output[/bold magenta]",
                                    title_align="",
                                    border_style="magenta",
                                    subtitle="Refiner Output")
                console.print(response_pnl)

                idx_cont = 0
                while refiner_response.usage.output_tokens > (agent.model.refine_max_tokens * 0.99):

                    zip_bytes = extract_output(refined_output, agent=agent, console=console, idx_cont=idx_cont)
                    idx_cont += 1
                    if idx_cont > 3:
                        break

                    console.print(f"[bold red]Warning truncated output, will try and continue ...[/bold red]")
                    refiner_continue_prompt = f"\n**PROMPT**\n\nContinuing from the Previous Response, please continue the response\n\n**PROMPT**\n\n"
                    refiner_continue_prompt += f"\n**Previous Response:**\n\n{refined_output}"
                    refiner_continue_prompt += f"\n\n**Original Query**\n\n{refiner_str}\n\n**Original Query**\n\n"

                    # response text
                    response_pnl = Panel(refiner_continue_prompt,
                                         title=f"[bold cyan]Continued Refiner Prompt[/bold cyan]",
                                         title_align="",
                                         border_style="cyan",
                                         subtitle="Continued Refiner Prompt")
                    console.print(response_pnl)

                    content = [{"type": "text", "text": refiner_continue_prompt}]
                    refiner_response = anthropic_client.messages.create(
                        model=agent.model.refiner_model,
                        max_tokens=agent.model.refine_max_tokens,
                        messages=[{"role": "user", "content": content}]
                    )

                    console.print(f"[bold green]Refined output, prompt length {len(refiner_str)}[/bold green]")
                    console.print(f"[bold green]Input Tokens {refiner_response.usage.input_tokens}[/bold green]")
                    console.print(f"[bold green]Output Tokens {refiner_response.usage.output_tokens}[/bold green]")
                    refined_output += refiner_response.content[0].text

                    # response text
                    response_pnl = Panel(refined_output,
                                         title=f"[bold blue]Continued Refiner Output[/bold blue]",
                                         title_align="",
                                         border_style="blue",
                                         subtitle="Continued Refiner Output")
                    console.print(response_pnl)

                break
            except RateLimitError as e:
                console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                idx_try += 1
                if idx_try > 3:
                    refined_output = "Rate Limit Error, anthropic AI sucks!"
                    break
                time.sleep(60)
    elif "gemini" in agent.model.refiner_model:
        model = genai.GenerativeModel(agent.model.refiner_model)
        ref_response = model.generate_content("".join(refiner_prompt))
        try:
            refined_output = ref_response.text
        except ValueError:
            # If the response doesn't contain text, check if the prompt was blocked.
            console.print(f"\n[bold red]Value Error During response.text[/bold red]")
            console.print(f"\n[bold red]Prompt Feedback : {ref_response.prompt_feedback}[/bold red]")
            console.print(f"\n[bold red]Finish Reason : {ref_response.candidates[0].finish_reason}[/bold red]")
            console.print(f"\n[bold red]Safety Ratings : {ref_response.candidates[0].safety_ratings}[/bold red]")
            refined_output = "come again?"
    elif 'gpt' in agent.model.refiner_model:
        raise ValueError(f"Unsupported refiner model: {agent.model.refiner_model}")
    else:
        raise ValueError(f"Unsupported refiner model: {agent.model.refiner_model}")


    response_pnl = Panel(refined_output,
                         title="[bold orange]Refined Result[/bold orange]",
                         border_style="white",
                         subtitle="Refined Result")
    console.print(response_pnl)

    return refined_output


def refine_output(agent: AgentConfig, idx_ref: int, era_output: str, console: Console):
    console.print("\n[bold]Refining the Subtask results[/bold]")

    subtask_str = '\n\n'.join([f"**Subtask {i}**\n{r}" for i, r in enumerate(agent.subtask_results[idx_ref])])
    refiner_prompt = [
        f"** Objective **\n\n{agent.objective}\n\n",
        ]
    if era_output is not None:
        refiner_prompt.append(f"** Baseline result **\n\n{era_output}\n\n",)

    refiner_folders = [
        f"** Subtask Results **\n\n{subtask_str}\n\n",
        "** PROMPT **\n\n",
        agent.model.refiner_prompt,
        *refiner_prompt,
        "Provide a relevent, brief and descriptive name for the project and include it in the final output in the format <project_name>name</project_name>. ",
        "INCLUDE THE FOLLOWING:\n",
        "1. Folder Structure: Provide the folder structure as a valid JSON object, ",
        "where each key represents a folder or file, and nested keys represent subfolders. ",
        "Use null values for files. Ensure the JSON is properly formatted without any syntax errors. ",
        "Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, "
        "separating items with commas as necessary. Wrap the JSON object in <folder_structure> tags.\n"
        ]
    refiner_str = "".join(refiner_folders + ["Do not include the file contents in this task, those will be generated in subsequent tasks.\n"])

    objective_pnl = Panel(agent.objective,
                          title="[bold orange]Original Objective[/bold orange]",
                          border_style="blue",
                          subtitle="Original Objective")
    console.print(objective_pnl)

    refined_output = None
    if "claude" in agent.model.refiner_model:
        idx_try = 0
        while True:
            console.print(f"[bold green]Refined output, prompt length "
                            f"{len(refiner_str)}[/bold green]")
            try:
                content = [{"type": "text", "text": refiner_str}]
                refiner_response = anthropic_client.messages.create(
                    model=agent.model.refiner_model,
                    max_tokens=agent.model.refine_max_tokens,
                    messages=[{"role": "user", "content": content}]
                )
                console.print(f"[bold green]Input Tokens "
                              f"{refiner_response.usage.input_tokens}"
                              "[/bold green]")
                console.print(f"[bold green]Output Tokens "
                              f"{refiner_response.usage.output_tokens}"
                              "[/bold green]")
                refined_output = refiner_response.content[0].text

                # response text
                response_pnl = Panel(refined_output, 
                                     title=f"[bold magenta]Refiner Output[/bold magenta]",
                                     title_align="",
                                     border_style="magenta",
                                     subtitle="Refined Folder Structure")
                console.print(response_pnl)

                idx_cont = 0
                while refiner_response.usage.output_tokens > (agent.model.refine_max_tokens * 0.99):
                    console.print(f"[bold red]Warning truncated output, will try and save result ...[/bold red]")
                    zip_bytes = extract_output(refined_output, agent=agent, console=console, idx_cont=idx_cont)

                    idx_cont += 1
                    if idx_cont > 3:
                        break
                    
                    refiner_continue_prompt = f"** PROMPT **\n\nContinuing from the Previous Response, please continue the response\n\n"
                    refiner_continue_prompt += f"** Previous Response **\n\n{refined_output}\n\n"
                    refiner_continue_prompt += f"** Original Query **\n\n{refiner_str}\n\n"

                    # response text
                    response_pnl = Panel(refiner_continue_prompt,
                                         title=f"[bold cyan]Continued Refiner Prompt[/bold cyan]",
                                         title_align="",
                                         border_style="cyan",
                                         subtitle="Continued Refiner Prompt")
                    console.print(response_pnl)

                    content = [{"type": "text", "text": refiner_continue_prompt}]
                    refiner_response = anthropic_client.messages.create(
                        model=agent.model.refiner_model,
                        max_tokens=agent.model.refine_max_tokens,
                        messages=[{"role": "user", "content": content}]
                    )

                    console.print(f"[bold green]Refined output, prompt length {len(refiner_str)}[/bold green]")
                    console.print(f"[bold green]Input Tokens {refiner_response.usage.input_tokens}[/bold green]")
                    console.print(f"[bold green]Output Tokens {refiner_response.usage.output_tokens}[/bold green]")
                    refined_output += refiner_response.content[0].text

                    # response text
                    response_pnl = Panel(refined_output,
                                         title=f"[bold blue]Continued Refiner Output[/bold blue]",
                                         title_align="",
                                         border_style="blue",
                                         subtitle="Continued Refiner Output")
                    console.print(response_pnl)

                break
            except RateLimitError as e:
                console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                idx_try += 1
                if idx_try > 2:
                    refined_output = "Rate Limit Error, anthropic AI sucks!"
                    break
                time.sleep(60)
        
        # Extract the folder structure and files
        folder_structure = None
        
        if "<folder_structure>" in refined_output:
            folder_structure = json.loads(refined_output.split("<folder_structure>")[1].split("</folder_structure>")[0])

            # extract the files
            files = {}
            def walk_folder(name, entry, files, folder_structure=refined_output, agent=agent):
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        walk_folder(f"{name}/{key}", value, files)
                else:
                    existing_files = "\n\n".join([f"{c}" for n, c in files.items()])
                    refiner_files = [
                        f"** Subtask Results **\n{subtask_str}",
                        f"** Folder Structure **\n{folder_structure}",
                        f"** Existing Files **\n\n{existing_files}",
                        "** PROMPT **",
                        agent.model.refiner_prompt,
                        f"Please include ONLY the file contents for {name} and not any other info!!",
                        f"DO NOT INCLUDE the triple backticks ``` and filetype just the text inside the files!",
                        ]
                    refiner_str = "\n\n".join(refiner_files)
                    idx_try = 0
                    while True:
                        try:
                            content = [{"type": "text", "text": refiner_str}]
                            refiner_response = anthropic_client.messages.create(
                                model=agent.model.refiner_model,
                                max_tokens=agent.model.refine_max_tokens,
                                messages=[{"role": "user", "content": content}]
                            )
                            console.print(f"[bold green]Refined output, prompt length "
                                          f"{len(refiner_str)}[/bold green]")
                            console.print(f"[bold green]Input Tokens "
                                          f"{refiner_response.usage.input_tokens}"
                                          "[/bold green]")
                            console.print(f"[bold green]Output Tokens "
                                          f"{refiner_response.usage.output_tokens}"
                                          "[/bold green]")
                            file_output = f'\n\n<file name="{name}">\n{refiner_response.content[0].text}\n</file>\n\n'
                            files[name] = file_output

                            # response text
                            response_pnl = Panel(file_output, 
                                                 title=f"[bold magenta]Refiner Output[/bold magenta]",
                                                 title_align="",
                                                 border_style="magenta",
                                                 subtitle=f"Refined File Output {name}")
                            console.print(response_pnl)

                            if refiner_response.usage.output_tokens > (agent.model.refine_max_tokens * 0.99):
                                console.print(f"[bold red]Warning truncated output, will try and save result ...[/bold red]")
                                zip_bytes = extract_output(refined_output, agent=agent, console=console, idx_cont=idx_try)

                            break
                        except RateLimitError as e:
                            console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                            idx_try += 1
                            if idx_try > 2:
                                refined_output = "Rate Limit Error, anthropic AI sucks!"
                                break
                            time.sleep(60)

            walk_folder("", folder_structure, files)
            for filename, content in files.items():
                refined_output += content


    elif "gemini" in agent.model.refiner_model:
        model = genai.GenerativeModel(agent.model.refiner_model)
        ref_response = model.generate_content("".join(refiner_prompt))
        try:
            refined_output = ref_response.text
        except ValueError:
            # If the response doesn't contain text, check if the prompt was blocked.
            console.print(f"\n[bold red]Value Error During response.text[/bold red]")
            console.print(f"\n[bold red]Prompt Feedback : {ref_response.prompt_feedback}[/bold red]")
            console.print(f"\n[bold red]Finish Reason : {ref_response.candidates[0].finish_reason}[/bold red]")
            console.print(f"\n[bold red]Safety Ratings : {ref_response.candidates[0].safety_ratings}[/bold red]")
            refined_output = "come again?"

        # Extract the folder structure and files
        folder_structure = None
        if "<folder_structure>" in refined_output:
            folder_structure = json.loads(refined_output.split("<folder_structure>")[1].split("</folder_structure>")[0])

            # extract the files
            files = {}
            def walk_folder(name, entry, files, refined_output=refined_output, agent=agent):
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        walk_folder(f"{name}/{key}", value, files)
                else:
                    existing_files = "\n\n".join([f"{c}" for n, c in files.items()])
                    refiner_files = [
                        f"** Subtask Results **\n{subtask_str}",
                        f"** Folder Structure **\n{folder_structure}",
                        f"** Existing Files **\n\n{existing_files}",
                        "** PROMPT **",
                        agent.model.refiner_prompt,
                        f"Please include ONLY the file contents for {name} and not any other info!!",
                        f"DO NOT INCLUDE the triple backticks ``` and filetype just the text inside the files!",
                        ]
                    refiner_str = "\n\n".join(refiner_files)
                    refiner_response = model.generate_content(refiner_str)
                    file_output = None
                    try:
                        file_output = ref_response.text
                    except ValueError:
                        # If the response doesn't contain text, check if the prompt was blocked.
                        console.print(f"\n[bold red]Value Error During response.text[/bold red]")
                        console.print(f"\n[bold red]Prompt Feedback : {ref_response.prompt_feedback}[/bold red]")
                        console.print(f"\n[bold red]Finish Reason : {ref_response.candidates[0].finish_reason}[/bold red]")
                        console.print(f"\n[bold red]Safety Ratings : {ref_response.candidates[0].safety_ratings}[/bold red]")
                        file_output = "come again?"
                    if f'<file name="{name}">' not in file_output:
                        file_output = f'\n\n<file name="{name}">\n{file_output}\n</file>\n\n'
                    else:
                        file_output = f'\n\n{file_output}\n\n'

                    files[name] = file_output

            walk_folder("", folder_structure, files)
            for filename, content in files.items():
                refined_output += content


    elif 'igpt' in agent.model.refiner_model:
        global igpt_client
        conversation = []
        role = "You are a master software architect."
        console.print(f"\n[bold]Generating File Structure[/bold]")
        console.print(f"[bold]iGPT content length: {len(refiner_str)}[/bold]")
        console.print(f"[bold]iGPT total length: {len(refiner_str) + len(role)}[/bold]")
        conversation.append({'role': 'system', 'content': role})
        conversation.append({'role': 'user', 'content': refiner_str})
        for idx_try in range(3):
            ref_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))
            if "Token has expired" in ref_response:
                igpt_client = iGPT(key=IGPT_KEY, secret=IGPT_SECRET)
                ref_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))

            if 'usage' in ref_response:
                console.print(f"[bold green]Refiner File Structure Tokens[/bold green]")
                console.print(f"[bold green]Input Tokens {ref_response['usage']['promptTokens']}[/bold green]")
                console.print(f"[bold green]Output Tokens {ref_response['usage']['completionTokens']}[/bold green]")
            else:
                console.print(f"[bold red]Error querying refinement model {ref_response}[/bold red]")
                continue
            if 'currentResponse' in ref_response:
                refined_output = ref_response['currentResponse']
                break
        if refined_output is None:
            refined_output = "come again?"

        # Extract the folder structure and files
        folder_structure = None
        if "<folder_structure>" in refined_output:
            folder_structure = json.loads(refined_output.split("<folder_structure>")[1].split("</folder_structure>")[0])

            # extract the files
            files = {}
            def walk_folder(name, entry, files, folder_structure=refined_output, agent=agent):
                global igpt_client
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        walk_folder(f"{name}/{key}", value, files)
                else:
                    existing_files = "\n\n".join([f"{c}" for n, c in files.items()])
                    refiner_files = [
                        f"** Subtask Results **\n{subtask_str}",
                        f"** Folder Structure **\n{folder_structure}",
                        f"** Existing Files **\n\n{existing_files}",
                        "** PROMPT **",
                        agent.model.refiner_prompt,
                        f"Please include ONLY the file contents for {name} and not any other info!!",
                        f"DO NOT INCLUDE the triple backticks ``` and filetype just the text inside the files!",
                        ]
                    file_output = None
                    conversation = []
                    role = "You are a master software architect."
                    refiner_file_str = "\n\n".join(refiner_files)
                    console.print(f"\n[bold]Generating File Output For : {name}[/bold]")
                    console.print(f"[bold]iGPT content length: {len(refiner_file_str)}[/bold]")
                    console.print(f"[bold]iGPT total length: {len(refiner_file_str) + len(role)}[/bold]")
                    role = "You are a expert at coding large projects who can comprehend lots of detail."
                    conversation.append({'role': 'system', 'content': role})
                    conversation.append({'role': 'user', 'content': refiner_file_str})

                    for idx_try in range(3):
                        file_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))
                        if "Token has expired" in file_response:
                            igpt_client = iGPT(key=IGPT_KEY, secret=IGPT_SECRET)
                            file_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))
                        if 'usage' in file_response:
                            console.print(f"[bold green]Refiner File Output Tokens[/bold green]")
                            console.print(f"[bold green]Input Tokens {file_response['usage']['promptTokens']}[/bold green]")
                            console.print(f"[bold green]Output Tokens {file_response['usage']['completionTokens']}[/bold green]")
                        else:
                            console.print(f"[bold red]Error querying refinement model {file_response}[/bold red]")
                            continue
                        if 'currentResponse' in file_response:
                            file_output = file_response['currentResponse']
                            break
                    if file_output is None:
                        file_output = "come again?"

                    file_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))
                    if 'usage' in file_response:
                        console.print(f"[bold green]Refiner output[/bold green]")
                        console.print(f"[bold green]Input Tokens {file_response['usage']['promptTokens']}[/bold green]")
                        console.print(f"[bold green]Output Tokens {file_response['usage']['completionTokens']}[/bold green]")
                    try:
                        file_output = file_response['currentResponse']
                    except TypeError as e:
                        console.print(f"[bold red]Error querying file contents {file_response}[/bold red]")
                        file_output = "come again?"

                    if f'<file name="{name}">' not in file_output:
                        file_output = f'\n\n<file name="{name}">\n{file_output}\n</file>\n\n'
                    else:
                        file_output = f'\n\n{file_output}\n\n'

                    files[name] = file_output

            walk_folder("", folder_structure, files)
            for filename, content in files.items():
                refined_output += content


    elif 'gpt' in agent.model.refiner_model:
        raise ValueError(f"Unsupported refiner model: {agent.model.refiner_model}")


    else:
        raise ValueError(f"Unsupported refiner model: {agent.model.refiner_model}")


    response_pnl = Panel(refined_output,
                         title="[bold orange]Refined Result[/bold orange]",
                         border_style="white",
                         subtitle="Final Refined Result")
    console.print(response_pnl)

    return refined_output


# ----------------------------------------------------------------------------
# Run the subagent
# ----------------------------------------------------------------------------


def run_subtask_agent(agent: AgentConfig, subtask_query: str, console: Console):

    if "claude" in agent.model.subagent_model:
        while True:
            try:
                subtask_messages = [{"role": "user", "content": [{"type": "text", "text": subtask_query}]}]
                subagent_response = anthropic_client.messages.create(
                    model=agent.model.subagent_model,
                    max_tokens=agent.model.sub_max_tokens,
                    messages=subtask_messages
                )
                console.print(f"[bold green]Subagent output prompt length {len(subtask_query)}[/bold green]")
                console.print(f"[bold green]Input Tokens {subagent_response.usage.input_tokens}[/bold green]")
                console.print(f"[bold green]Output Tokens {subagent_response.usage.output_tokens}[/bold green]")
                idx_cont = 0
                subtask_result = subagent_response.content[idx_cont].text

                while subagent_response.usage.output_tokens > (agent.model.sub_max_tokens * 0.99):
                    response_pnl = Panel(subtask_result,
                                         title="[bold orange]Incremental SubAgent Result[/bold orange]",
                                         border_style="red",
                                         subtitle="[bold orange]Incremental SubAgent Result[/bold orange]")
                    console.print(response_pnl)

                    subagent_continue_prompt = f"\n\n*** PROMPT ***\n\n"
                    subagent_continue_prompt += f"\n\n*** Continuing from the Previous Response and fulfilling the Original Query, please continue the response ***\n\n"
                    subagent_continue_prompt += f"\n\n** Previous Response **\n\n{subtask_result}"
                    subagent_continue_prompt += f"\n\n** Original Query **\n\n{subtask_query}"
                    subtask_messages = [{"role": "user", "content": [{"type": "text", "text": subagent_continue_prompt}]}]
                    console.print("[bold red]Warning truncated output, will try and continue ...[/bold red]")
                    subprompt_pnl = Panel(subagent_continue_prompt,
                                          title="[bold orange]Incremental SubAgent Prompt[/bold orange]",
                                          border_style="red",
                                          subtitle="[bold orange]Incremental SubAgent Prompt[/bold orange]")
                    console.print(subprompt_pnl)
                    subagent_response = anthropic_client.messages.create(
                        model=agent.model.subagent_model,
                        max_tokens=agent.model.sub_max_tokens,
                        messages=subtask_messages
                    )
                    console.print(f"[bold green]Subagent output prompt length {len(subagent_continue_prompt)}[/bold green]")
                    console.print(f"[bold green]Input Tokens {subagent_response.usage.input_tokens}[/bold green]")
                    console.print(f"[bold green]Output Tokens {subagent_response.usage.output_tokens}[/bold green]")
                    console.print(f"[bold green]Contents Length {len(subagent_response.content)}[/bold green]")
                    subtask_result += subagent_response.content[idx_cont].text
                break
            except RateLimitError as e:
                console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                time.sleep(60)


    elif 'gemini' in agent.model.subagent_model:
        # TODO: need to check this query and output tokens etc for google model
        model = genai.GenerativeModel(agent.model.subagent_model)
        subtask_prompt = f"**prompt:**\n\n{subtask_query}\n\n"
        subagent_response = model.generate_content("".join(subtask_prompt), safety_settings=ggl_safety_settings)
        try:
            subtask_result = subagent_response.text
        except ValueError:
            # If the response doesn't contain text, check if the prompt was blocked.
            console.print(f"\n[bold red]Value Error During response.text[/bold red]")
            console.print(f"\n[bold red]Prompt Feedback : {subagent_response.prompt_feedback}[/bold red]")
            console.print(f"\n[bold red]Finish Reason : {subagent_response.candidates[0].finish_reason}[/bold red]")
            console.print(f"\n[bold red]Safety Ratings : {subagent_response.candidates[0].safety_ratings}[/bold red]")
            subtask_result = "come again?"


    elif 'igpt' in agent.model.subagent_model:
        global igpt_client
        conversation = []
        subtask_prompt = f"**prompt:**\n\n{subtask_query}\n\n"
        role = "You are coding expert sub-agent who knowns about semiconductor physical design tasks."
        console.print(f"\n[bold]iGPT content length: {len(subtask_prompt)}[/bold]")
        console.print(f"\n[bold]iGPT total length: {len(subtask_prompt) + len(role)}[/bold]")
        conversation.append({'role': 'system', 'content': role})
        conversation.append({'role': 'user', 'content': subtask_prompt})
        subagent_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))

        if "Token has expired" in subagent_response:
            igpt_client = iGPT(key=IGPT_KEY, secret=IGPT_SECRET)
            subagent_response = igpt_client.generate(conversation=conversation, correlationId=str(agent.id))
 
        if 'usage' in subagent_response:
            console.print(f"[bold green]Orchestrator output[/bold green]")
            console.print(f"[bold green]Input Tokens {subagent_response['usage']['promptTokens']}[/bold green]")
            console.print(f"[bold green]Output Tokens {subagent_response['usage']['completionTokens']}[/bold green]")
        try:
            subtask_result = subagent_response['currentResponse']
        except TypeError as e:
            console.print(f"[bold red]Error querying subagent model {subagent_response}[/bold red]")
            subtask_result = "come again?"


    else:
        raise ValueError(f"Unsupported subagent model: {agent.model.subagent_model}")

    response_pnl = Panel(subtask_result,
                         title="[bold orange]SubAgent Result[/bold orange]",
                         border_style="white",
                         subtitle="SubAgent Result")
    console.print(response_pnl)

    return subtask_result


# --------------------------------------------------------------------------------
# Function for generating the combined query for the subagent
# --------------------------------------------------------------------------------


def generate_subtask_prompt(agent: AgentConfig, orch_response: str,
                            search_query: str, era_output: str,
                            idx_ref: int, idx_task: int, 
                            console: Console):

    # create a subtask query
    system_message = ""
    subtask_query = ""
    if idx_ref != 0:
        system_message = "\n** Baseline Result **\n"
        system_message += f"{era_output}\n\n"
    if idx_task != 0:
        res = [f"**Task Result {idx}**\n{result}"
               for idx, result in enumerate(agent.subtask_results[idx_ref])]
        system_message = "\n** Previous Task Results **\n"
        system_message += "\n".join(res)
    subtask_query += orch_response
    subtask_query += system_message

    # check if files are included
    if (idx_ref == 0) and (idx_task == 0) and len(agent.files) > 0:
        subtask_query += "** FILES **\n\n" + "\n".join([f'** File content ({name}) **\n{cont}\n\n' for name, cont in agent.files.items()])

    # add in the search query if needed
    search_result = None
    if agent.use_search and search_query is not None:
        search_result = query_search_provider(query=search_query, provider="tavily", console=console)
        subtask_query += f"\n** Search Results **\n{search_result}"

    subtask_query += f"\n\nONLY INCLUDE THE CONCISE AND COMPLETE REPSPONSE TO THE SUBTASK IN THIS STEP!!\n\n"

    return subtask_query


# --------------------------------------------------------------------------------
# Run the orchestrator to complete the objective
# --------------------------------------------------------------------------------


def run_orchestrator_loop(agent: AgentConfig, console: Console=Console(record=True)):
    console.print("\n[bold]Starting orchestrator loop[/bold]")
    console.print(f"[green]Strategy : {agent.model.strategy}[/green]")
    console.print(f"[green]Orchestrator : {agent.model.orchestrator_model}[/green]")
    console.print(f"[green]Subagent : {agent.model.subagent_model}[/green]")
    console.print(f"[green]Refiner : {agent.model.refiner_model}[/green]")

    # refresh the bear token
    global igpt_client
    igpt_client = iGPT(IGPT_KEY, IGPT_SECRET)

    era_output = None
    orch_response = None
    for idx_ref in range(agent.model.refine_iter):
        console.print(f"\n[bold]Refinment Iteration {idx_ref + 1}[/bold]")
        agent.subtask_queries[idx_ref] = []
        agent.subtask_results[idx_ref] = []

        for idx_task in range(agent.model.task_iter):

            console.print(f"\n[bold]Running Orchestrator for SubTask Prompt: Refine Iteration {idx_ref + 1} Task Iteration {idx_task + 1}[/bold]")
            if (idx_ref == 0) and (idx_task == 0):
                agent.include_files = True
                (
                    orch_response,
                    search_query
                ) = query_orchestrator(agent, idx_ref, era_output, console=console)
                agent.include_files = False
            else:
                (
                    orch_response,
                    search_query
                ) = query_orchestrator(agent, idx_ref, era_output, console=console)

            if "Objective Complete:" in orch_response:
                break

            subtask_query = generate_subtask_prompt(agent, orch_response,
                                                    search_query, era_output,
                                                    idx_ref, idx_task, console=console)
            subtask_result = run_subtask_agent(agent, subtask_query, console=console)

            agent.subtask_queries[idx_ref].append(subtask_query)
            agent.subtask_results[idx_ref].append(subtask_result)

        if orch_response is not None and "Objective Complete:" in orch_response:
            break

        # summarize the results for this era
        era_output = refine_output(agent, idx_ref, era_output, console=console)
        agent.era_results.append(era_output)

    # Call the refiner
    if orch_response is not None and "Objective Complete:" in orch_response:
        final_output = refine_output(agent, idx_ref, era_output, console=console)
    else:
        final_output = era_output

    # Process the final output
    zip_bytes = extract_output(final_output, agent=agent, console=console)

    return zip_bytes


# --------------------------------------------------------------------------------
# Extract the final output into a zip file
# --------------------------------------------------------------------------------


def extract_output(refined_output: str, agent: AgentConfig, console: Console, idx_cont: int = None):
    console.print("\n[bold]Extracting the final output[/bold]")

    # extract the project name
    if '<project_name>' in refined_output:
        project_name = f'{refined_output.split("<project_name>")[1].split("</project_name>")[0]}_{agent.id}'
    else:
        project_name = f"{agent.name}_{agent.id}"
    if idx_cont is not None:
        project_name = f"{project_name}_{idx_cont}"

    # print project name
    console.print(f"[green]Project Name : {project_name}[/green]")

    # write the final output
    with open(f"final/final_output_{project_name}.md", "w") as f:
        f.write(refined_output)

    console.print(f"[green]Download Project http://{os.environ['HOSTNAME']}:{os.environ['APP_PORT']}/download_project/{agent.id}[/green]")

    console.print(f"[green]Successfully Completed Objective :-) [/green]")
    console.print(f"[green]Name {agent.name}[/green]")
    console.print(f"[green]Project {project_name}[/green]")
    console.print(f"[green]Refined Output Length {len(refined_output)}[/green]")
    console.print(f"[green]Done :)[/green]")
    console.print(f"[green]Done :)[/green]")
    console.print(f"[green]Done :)[/green]")
    console.print(f"[green]Done :)[/green]")

    # extract the folder structure
    folder_structure = None
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:

        if "<folder_structure>" not in refined_output:
            console.print(f"[red]Folder Structure Not Found In Output[/red]")
        else:
            folder_structure = json.loads(refined_output.split("<folder_structure>")[1].split("</folder_structure>")[0])

            # extract the files
            files = {}
            def walk_folder(name, entry, files):
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        walk_folder(f"{name}/{key}", value, files)
                else:
                    if f'<file name="{name}">' not in refined_output:
                        console.print(f"\n[bold red]Missing file contents for {name}[/bold red]")
                        return None
                    file_contents = refined_output.split( f'<file name="{name}">' )[1].split("</file>")[0]
                    files[name] = file_contents

            walk_folder("", folder_structure, files)

            for file_name, data in files.items():
                zip_file.writestr(file_name[1:], data)
            
        zip_file.writestr("folder_structure.json", json.dumps(folder_structure, indent=4))
        zip_file.writestr("final_output.txt", refined_output)
        zip_file.writestr("exec_log.html", console.export_html())

    with open(f'./output/{project_name}.zip', 'wb') as f:
        f.write(zip_buffer.getvalue())
    with open(f'./output/{agent.id}_final.zip', 'wb') as f:
        f.write(zip_buffer.getvalue())

    return zip_buffer.getvalue()


# --------------------------------------------------------------------------------
# Run orchestrator as script
# --------------------------------------------------------------------------------


def run_agentapp():
    #objective = input("Enter the objective : ")class ModelConfig(BaseModel):
    objective = """
Please take your time to think before answering and use as much detail as needed to complete the objective!!
Build a web app using fastapi, css and html allows user to run AI agent system.
Include detailed documentaion on how to install, run and contribute.
Structure each session's data as a pydandic BaseModel for easy storing and
retrieval in a database. Allow the user to select the orchestrator, subagent
and refiner model versions. Search the web to make sure you include all
the latest and most relevant AI models. The theme should be dark and user
interface calm and relaxing and using best pratices for web design. The user
should provide an objective, make the input for the objective take up most of
the page. Use mongo db and motor asynchrounous library to access for storing
the configs. Include an requirements.txt file and use pytest for testing.
I think you are going to do an awesome job at this!!!

Here are the pydantic models for agent and model config

```python

from typing import Any
from typing import Dict
from typing import Annotated
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import AfterValidator
from pydantic import WithJsonSchema

from bson import ObjectId

def validate_object_id(v: Any) -> ObjectId:
    if isinstance(v, ObjectId):
        return v
    if ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")


PyObjectId = Annotated[
    Union[str, ObjectId],
    AfterValidator(validate_object_id),
    PlainSerializer(lambda x: str(x), return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


class ModelConfig(BaseModel):
    orchestrator_model: str = Field(...)
    refiner_model: str = Field(...)
    subagent_model: str = Field(...)
    orchestrator_prompt: str = orch_base_prompt
    refiner_prompt: str = refiner_base_prompt
    strategy: str = Field(...)
    task_iter: int = 5
    refine_iter: int = 3
    orch_max_tokens: int = 4096
    sub_max_tokens: int = 4096
    refine_max_tokens: int = 4096


class AgentConfig(BaseModel):
    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    name: str = Field(...)
    objective: str = Field(...)
    subtask_queries: dict[int, list[str]] = {}
    subtask_results: dict[int, list[str]] = {}
    era_results: list[str] = []
    files: dict[str, str] = {}
    use_search: bool = False
    include_files: bool = False
    model: ModelConfig

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

```
"""

    '''
    claude-3-opus-20240229
    gemini-1.5-pro-latest
    claude-3-haiku-20240307
    claude-3-sonnet-20240229
    '''
    model = ModelConfig(orchestrator_model="igpt4-turbo",
                        refiner_model="igpt4-turbo",
                        subagent_model="igpt4-turbo",
                        task_iter=3,
                        refine_iter=4,
                        strategy="IterativeRefinement")
    agent = AgentConfig(name='AI Agent App', objective=objective, model=model)

    zip_bytes = run_orchestrator_loop(agent)


def run_fc_debugger():
    #objective = input("Enter the objective : ")class ModelConfig(BaseModel):
    objective = """
Please take your time to think before answering and use as much detail as needed to complete the objective!!
Build a web app using fastapi, css and html allows user to debug a python script.
The interface should be a split screen with the script and the console.
The user should be able to edit the script and step through the lines of code one by one or run until hitting a breakpoint.
If the code execution is halted the user can enter commands into the console to be evaluated.
Use html templates to display the window.
Implement code folding and syntax highlighting for the script window, this will provide a great user inferface.
Include an requirements.txt file. and use pytest for the testing.
Make sure to allow the script window and console window to scroll separately!
Don't use node or npm, just use javascript and html, link to the codemirror libraries in CDN and don't include the files themselves.
Use a relaxing dark theme for the app and generate calm feelings to help the user focus and debug their code.
You are great at coding, I know you'll do a great job:-)
DO NOT USE PDB OR DEBUGGER, ONLY SEND ONE LINE AT A TIME TO THE SUBPROCESS UNTIL YOU HIT A BREAKPOINT OR VIEW AN ERROR IN THE OUTPUT.
For interacting with the subprocess use the below python code, DO NOT USE PDB, ONLY INTERACT WITH THE SUBPROCESS AS BELOW by sending a line at a time!!!


```python
from subprocess import Popen, PIPE, STDOUT

shell_command = ["python"]
subprocess = Popen(shell_command, stdout=PIPE, stdin=PIPE, stderr=STDOUT)

filename = "code_file.py"
with open(filename) as fid:
    code_lines = fid.read_lines()

for line in code_lines:
    subprocess.communicate(line.encode())[0].rstrip()
    result = subprocess.stdout.readline().rstrip()
    print(result)

```
"""

    '''
    claude-3-opus-20240229
    gemini-1.5-pro-latest
    claude-3-haiku-20240307
    claude-3-sonnet-20240229
    '''
    model = ModelConfig(orchestrator_model="claude-3-sonnet-20240229",
                        refiner_model="claude-3-sonnet-20240229",
                        subagent_model="gemini-1.5-pro-latest",
                        task_iter=5,
                        refine_iter=4,
                        sub_max_tokens=8192,
                        #refine_max_tokens=8192,
                        strategy="IterativeRefinement")
    agent = AgentConfig(name='Fusion Compiler Python Debugger', objective=objective, model=model)

    zip_bytes = run_orchestrator_loop(agent)



if __name__ == "__main__":
    run_agentapp()


# --------------------------------------------------------------------------------
# Done :)
# --------------------------------------------------------------------------------
