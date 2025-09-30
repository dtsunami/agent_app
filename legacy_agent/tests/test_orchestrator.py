# --------------------------------------------------------------------------------
# File : test_orchestrator.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : Prompts and strategies for AI Agent orchestration
# Purp : The base prompts and iteration strategies for the
#        AI Agent application.
# --------------------------------------------------------------------------------

from orchestrator import ModelConfig, AgentConfig
from orchestrator import query_orchestrator, run_orchestrator_loop
from orchestrator import extract_output


def test_orchestrator_query_claude_opus():

    objective = "Create a report on the current state of the US economy. "
    objective += "Include risks and disruptive factors and "
    objective += "provide recommendations for the next 3 months. "
    objective += "Use the most recent data available and spare no detail. "
    model = ModelConfig(orchestrator_model="claude-3-opus-20240229",
                        refiner_model="claude-3-opus-20240229",
                        subagent_model="claude-3-haiku-20240307",
                        task_iter=2,
                        refine_iter=3,
                        strategy="FixedPointIteration")
    agent = AgentConfig(objective=objective,
                        use_search=True,
                        include_files=False,
                        model=model)
    result, search_query = query_orchestrator(agent)

    assert result is not None
    assert search_query is not None


def test_orchestrator_loop_claude_opus():

    objective = "Create a webapp with at least 100 dessert recipes. "
    objective += "Include features for filtering what features we want in the dessert as well as a search engine and a list of ingredients. "
    objective += "Use a rainbow theme and store all the desserts in a database, and make sure to populate the database with at least 100 recipes. "
    objective += "Allow users to log in and create a profile so that they can favorite recipes and comment on the recipes that they like. "
    objective += "Make sure you use the best practices and security standard and moderation of toxic posts using AI. "
    model = ModelConfig(orchestrator_model="claude-3-sonnet-20240229",
                        refiner_model="claude-3-opus-20240229",
                        subagent_model="claude-3-sonnet-20240229",
                        task_iter=2,
                        refine_iter=3,
                        strategy="FixedPointIteration")
    agent = AgentConfig(objective=objective,
                        use_search=True,
                        include_files=False,
                        model=model)

    zip_bytes = run_orchestrator_loop(agent)

    assert final_output is not None
    assert zip_bytes is not None


def test_orchestrator_loop_claude_3p5():

    objective = """
Create a web app using codemirror javascript library that allows the user to debug a python-like subprocess.

Use FastAPI, css, javascript and html templates using jinja2.

The subprocess should be spawned using pexpect library.

Use a dark and relaxing theme for the app and display the line numbers for code.

Include controls for setting and clearing breakpoints and running, continuing or stepping through the code.

The user should be able to provide the command line arguments and set environment variables for the subprocess.

Don't worry about security right now since this is proof of concept only.
"""
    model = ModelConfig(orchestrator_model="claude-3-5-sonnet-20240620",
                        refiner_model="claude-3-5-sonnet-20240620",
                        subagent_model="claude-3-5-sonnet-20240620",
                        task_iter=3,
                        refine_iter=2,
                        strategy="IterativeRefinement")
    agent = AgentConfig(name="PyBugger_v3p5",
                        objective=objective,
                        use_search=True,
                        include_files=False,
                        model=model)

    zip_bytes = run_orchestrator_loop(agent)

    assert final_output is not None
    assert zip_bytes is not None


def test_output_pybuddy():
    filepath = 'output/PyBuddy/final_output.txt'
    with open(filepath) as fid:
        final_output = fid.read()
    
    zip_bytes = extract_output(final_output)
    
    assert zip_bytes is not None