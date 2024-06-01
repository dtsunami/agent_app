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
