# --------------------------------------------------------------------------------
# File : vectordb.py
# Auth : Dan Gilbert
# Date : 6/2/2024
# Desc : This file contains code to index files into a vectordb.
# Purp : Used as a part of RAG(Retrieval augmented generation) ai agent system
# --------------------------------------------------------------------------------

import os
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore

POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
POSTGRES_HOST = os.environ['POSTGRES_HOST']
POSTGRES_PORT = int(os.environ['POSTGRES_PORT'])
POSTGRES_DB = os.environ['POSTGRES_DB']
POSTGRES_USER = os.environ['POSTGRES_USER']


# --------------------------------------------------------------------------------
# Index the contents of a directory and store it in the vector db
# --------------------------------------------------------------------------------


def store_folder_as_table(dirpath: str, table_name: str, embed_dim: int = 1536):

    documents = SimpleDirectoryReader(dirpath).load_data()

    vector_store = PGVectorStore.from_params(
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        password=POSTGRES_PASSWORD,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        table_name=table_name,
        embed_dim=embed_dim,  # openai embedding dimension = 1536
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    query_engine = index.as_query_engine()

    return query_engine


#---------------------------------------------------------------------------------
# Get an existing table's query engine to get info
#---------------------------------------------------------------------------------


def get_query_engine(table_name: str, embed_dim: int = 1536):
    vector_store = PGVectorStore.from_params(
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        password=POSTGRES_PASSWORD,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        table_name=table_name,
        embed_dim=embed_dim,  # openai embedding dimension = 1536
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine()

    return query_engine


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
