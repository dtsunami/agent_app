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

import re
import json
import asyncio

from datetime import datetime

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from agents import openai_client, anthropic_client, tavily_client, genai

from anthropic import RateLimitError

#---------------------------------------------------------------------------------
# Agent Configuration Models
#---------------------------------------------------------------------------------


class ModelConfig(BaseModel):
    orchestrator: str
    refiner: str
    subagent: str
    strategy: str
    max_iter: int
    orch_max_tokens: int
    sub_max_tokens: int
    refine_max_tokens: int

class AgentConfig(BaseModel):
    objective: str
    subtask_results: list[tuple[str, str]]
    files: dict[str, str]
    use_search: bool
    include_files: bool
    model: ModelConfig


#---------------------------------------------------------------------------------
# Initialize the console
#---------------------------------------------------------------------------------

console = Console()


#---------------------------------------------------------------------------------
# Query the orchestrator for the next task
# async 
#---------------------------------------------------------------------------------


def query_orchestrator(agent: AgentConfig):
    console.print(f"\n[bold]Query orchestrator model: {agent.model.orchestrator}[/bold]")
    
    results = [result for _, result in agent.subtask_results]
    results_str = "None"
    if len(results) > 0:
        results_str = "\n".join(results)
    
    orch_prompt = [
        "Prioritizing the following objective above all else, ",
        f"{' and using the file content ' if agent.include_files else ''}",
        f"{' without forgetting anything from the previous sub-task results, ' if len(results_str) > 0 else ''}",
        "please select the next sub-task that most advances the obective ",
        "and create a clear, encouraging and comprehensive prompt for a subagent to execute that sub-task. ",
        "ALWAYS CHECK CODE FOR ERRORS AND USE THE BEST PRACTICES FOR CODING TASKS AND INCLUDE FIXES FOR THE NEXT SUB-TASK.",
        "If you have any sugestions on how code can be improved or refactored, please include them in the next sub-task prompt. ",
        "Assess if the objective has been fully achieved and if not, break it down into the next sub-task. ",
        "If the previous sub-task results comprehensively complete all the requirements of the objective, ",
        "start your response with the phrase 'Objective Complete:'. ",
        "If the objective is not yet fully achieved, select the next most appropriate sub-task ",
        "and create a clear and comprehensive prompt for a subagent to execute that task.",
        f"\n\nObjective: {agent.objective}\n\n",
    ]
    if agent.include_files:
        orch_prompt += [f'File content ({name}) :\\n{cont}\n\n' for name, cont in agent.files.items()]
    
    messages = [{"role": "user", "content": [{"type": "text", "text": "".join(orch_prompt)}]}]

    if agent.use_search:
        # TODO: rewrite the boilerplate search query
        search_query = [
            "Please also generate a JSON object containing a single 'search_query' key, ",
            "which represents a question that, when asked online, would yield important information for solving the subtask. ",
            "The question should be specific and targeted to elicit the most relevant and helpful resources. ",
            "Format your JSON like this, with no additional text before or after:\n{'search_query': '<question>'}\n"
        ]
        messages[0]["content"].append({"type": "text", "text": "".join(search_query)})
    
    orch_response = None
    if 'claude' in agent.model.orchestrator:
        while True:
            try:
                orch_response = anthropic_client.messages.create(
                    model=agent.model.orchestrator,
                    max_tokens=agent.model.orch_max_tokens,
                    messages=messages
                )
                response_text = orch_response.content[0].text
                break
            except RateLimitError as e:
                console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                time.sleep(60)

    elif 'gemini' in agent.model.orchestrator:
        model = genai.GenerativeModel(agent.model.orchestrator)
        orch_response = model.generate_content("".join(orch_prompt))
        response_text = orch_response.text
    elif 'gpt' in agent.model.orchestrator:
        raise NotImplementedError("GPT-4 is not yet supported")
    else:
        raise ValueError(f"Unsupported orchestrator model: {agent.model.orchestrator}") 
    
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
    

#---------------------------------------------------------------------------------
# Search current data for the next task
#------------orch_prompt---------------------------------------------------------------------
from requests.exceptions import HTTPError

def query_search_provider(query: str, provider: str = "tavily"):
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



#---------------------------------------------------------------------------------
# Query the refiner for final output
#----------------------------------------------------------------idx_try
def refine_output(agent: AgentConfig):
    console.print(f"\n[bold]Refining the final output[/bold]")

    subtask_str = '\n'.join([r for _, r in agent.subtask_results])
    refiner_prompt = [
        f"Objective: {agent.objective}\n\n",
        f"Sub-task results:\n{subtask_str}\n\n",
        "Please review the sub-task results and refine them into a cohesive final output. ",
        "Add any missing documenation or details as needed. ",
        "Provide a relevent, brief and descriptive name for the project and include it in the final output in the format <project_name>name</project_name>. ",
        "INCLUDE THE FOLLOWING:\n",
        "1. Folder Structure: Provide the folder structure as a valid JSON object, ",
        "where each key represents a folder or file, and nested keys represent subfolders. ",
        "Use null values for files. Ensure the JSON is properly formatted without any syntax errors. ",
        "Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, "
        "separating items with commas as necessary. Wrap the JSON object in <folder_structure> tags.\n",
        "2. Code, Data or Image Files: For each  file, include ONLY THE FILE NAME AND NOT THE PATH, wrap the file contents in tags like this <file name='filename'>contents</file>. ",
        ]
    messages = [{"role": "user", "content": [{"type": "text", "text": "".join(refiner_prompt)}]}]

    if "claude" in agent.model.refiner:
        idx_try = 0
        while True:
            try:
                refiner_response = anthropic_client.messages.create(
                    model=agent.model.refiner,
                    max_tokens=agent.model.refine_max_tokens,
                    messages=messages
                )
                refined_output = refiner_response.content[0].text
                break
            except RateLimitError as e:
                console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                idx_try += 1
                if idx_try > 3:
                    break
                time.sleep(60)
    elif "gemini" in agent.model.refiner:
        model = genai.GenerativeModel(agent.model.refiner)
        ref_response = model.generate_content("".join(refiner_prompt))
        refined_output = ref_response.text
    elif 'gpt' in agent.model.refiner:
        raise ValueError(f"Unsupported refiner model: {agent.model.refiner}")
    else:
        raise ValueError(f"Unsupported refiner model: {agent.model.refiner}")

    return refined_output


#---------------------------------------------------------------------------------
# Run the orchestrator to complete the objective
#---------------------------------------------------------------------------------

import time

def run_orchestrator_loop(agent: AgentConfig):
    console.print("\n[bold]Starting orchestrator loop[/bold]")
    console.print(f"[green]Strategy : {agent.model.strategy}[/green]")
    console.print(f"[green]Orchestrator : {agent.model.orchestrator}[/green]")
    console.print(f"[green]Subagent : {agent.model.subagent}[/green]")
    console.print(f"[green]Refiner : {agent.model.refiner}[/green]")

    for idx in range(agent.model.max_iter):
        console.print(f"\n[bold]Iteration {idx + 1}[/bold]")

        if idx == 0:
            agent.include_files = True
            orch_response, search_query = query_orchestrator(agent)
            agent.include_files = False
        else:
            orch_response, search_query = query_orchestrator(agent)
        
        if "Objective Complete:" in orch_response:
            break
 
        # create a subtask query
        system_message = ""
        subtask_query = ""
        if idx != 0:
            system_message = "Previous Subagent tasks:\n"
            system_message += "\n".join(f"Task: {task}\nResult: {result}" for task, result in agent.subtask_results)
        subtask_query += orch_response

        # check if files are included
        if idx == 0 and len(agent.files) > 0:
            subtask_query += "\n\n" + "\n".join([f'File content ({name}) :\\n{cont}\n\n' for name, cont in agent.files.items()])
        
        # initial collateral without the seach included
        subtask_messages = [{"role": "user", "content": [{"type": "text", "text": subtask_query}]}]

        # add in the search query if needed
        search_result = None
        if agent.use_search and search_query is not None:
            search_result = query_search_provider(search_query)
            subtask_messages[0]["content"].append({"type": "text", "text": f"\nSearch Results:\n{search_result}"})
        
        # call the subagent
        if "claude" in agent.model.subagent:
            while True:
                try:
                    subagent_response = anthropic_client.messages.create(
                        model=agent.model.subagent,
                        max_tokens=agent.model.sub_max_tokens,
                        messages=subtask_messages,
                        system=system_message
                    )
                    subtask_result = subagent_response.content[0].text
                    break
                except RateLimitError as e:
                    console.print(f"\n[bold red]Hit Rate Limit Error, will retry in 60s[/bold red]")
                    time.sleep(60)
        elif 'gemini' in agent.model.subagent:
            model = genai.GenerativeModel(agent.model.subagent)
            subtask_prompt = f"prompt:\n {subtask_query}\n\n{system_message}\n\n"
            if search_result is not None:
                subtask_prompt += f"Search Results: {search_result}"
            subagent_response = model.generate_content("".join(subtask_prompt))
            subtask_result = subagent_response.text
        else:
            raise ValueError(f"Unsupported subagent model: {agent.model.subagent}")
        
        #if subagent_response.usage.output_tokens >= agent.model.sub_max_tokens:
        #    console.print(f"\n[bold red]Subagent response exceeded max tokens[/bold red]")
        #    # TODO: should recover from output truncation?
        
        agent.subtask_results.append((subtask_query, subtask_result))
    
    # Call the refiner
    final_output = refine_output(agent)


    # Process the final output
    zip_bytes = extract_output(final_output)

    return final_output, zip_bytes


#---------------------------------------------------------------------------------
# Extract the final output into a zip file
#---------------------------------------------------------------------------------


def extract_output(refined_output: str):
    console.print(f"\n[bold]Extracting the final output[/bold]")

    # extract the project name
    project_name = refined_output.split("<project_name>")[1].split("</project_name>")[0]
    console.print(f"[green]Project Name : {project_name}[/green]")

    with open(f"final_output{project_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md", "w") as f:
        f.write(refined_output)
    
    # extract the folder structure
    folder_structure = None
    if "<folder_structure>" in refined_output:
        folder_structure = json.loads(refined_output.split("<folder_structure>")[1].split("</folder_structure>")[0])
        
        # extract the files
        files = {}
        def walk_folder(name, entry, files):
            if isinstance(entry, dict):
                for key, value in entry.items():
                    walk_folder(key, value, files)
            else:
                if f"<file name='{name}'>" not in refined_output:
                    console.print(f"\n[bold red]Missing file contents for {name}[/bold red]")
                    return None
                file_contents = refined_output.split(f"<file name='{name}'>")[1].split("</file>")[0]
                files[name] = file_contents
        
        walk_folder(project_name, folder_structure, files)
    
        import io
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_name, data in files.items():
                zip_file.writestr(file_name, data)
            zip_file.writestr("folder_structure.json", json.dumps(folder_structure, indent=4))
            zip_file.writestr("final_output.txt", refined_output)

                
        with open(f'./output/{project_name}.zip', 'wb') as f:
            f.write(zip_buffer.getvalue())

        return zip_buffer.getvalue()
    else:
        return None 
    

#---------------------------------------------------------------------------------
# Run orchestrator as script
#---------------------------------------------------------------------------------


def run():
    #objective = input("Enter the objective : ")
    objective = "Build a web app using fastapi, css and html allows user to run AI agent system. "
    objective += "Include detailed documentaion on how to install, run and contribute. "
    objective += "Structure each session's data as a pydandic BaseModel for easy storing and retrieval in a database. "
    objective += "Allow the user to select the orchestrator, sub-agent and refiner model versions. "
    objective += "Search the web to make sure you include all the latest and most relevant AI models. "
    objective += "The theme should be dark and user interface calm and relaxing and using best pratices for web design. "
    objective += "The user should provide an objective, make the input for the objective take up most of the page. "
    

    model = ModelConfig(orchestrator="gemini-1.5-pro",
                        refiner="gemini-1.5-pro",
                        subagent="gemini-1.5-pro",
                        orch_max_tokens=4096,
                        sub_max_tokens=4096,
                        refine_max_tokens=4096,
                        max_iter=30,
                        strategy="FixedPointIteration")
    agent = AgentConfig(objective=objective,
                        subtask_results=[],
                        files={},
                        use_search=True,
                        include_files=False,
                        model=model)
    result, search_query = run_orchestrator_loop(agent)
    

if __name__ == "__main__":
    run()

#---------------------------------------------------------------------------------
# Done :)orch_response
#---------------------------------------------------------------------------------

