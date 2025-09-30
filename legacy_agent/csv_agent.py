# --------------------------------------------------------------------------------
# File : csv_agent.py
# Auth : Dan Gilbert
# Date : 6/2/2024
# Desc : This file contains code to index files into a vectordb.
# Purp : Used as a part of RAG(Retrieval augmented generation) ai agent system
# --------------------------------------------------------------------------------

import os

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI


# --------------------------------------------------------------------------------
# Convert csv file to query
# --------------------------------------------------------------------------------


def make_csv_agent(filepath: str, model_name: str):

    agent = create_csv_agent(
        OpenAI(temperature=0, model=model_name),
        filepath,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent


# --------------------------------------------------------------------------------
# Try it
# --------------------------------------------------------------------------------


def main():

    #query_engine = store_folder_as_table('./final', 'ai_agent_iterations')

    query_engine = get_query_engine('ai_agent_iterations')

    response = query_engine.query("Why do I need an AI agent system?")

    print(response)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------
# Done :)
# --------------------------------------------------------------------------------
