#---------------------------------------------------------------------------------
# File : test_server.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : Prompts and strategies for AI Agent orchestration
# Purp : The base prompts and iteration strategies for the AI Agent application.
#---------------------------------------------------------------------------------

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "<h1>Welcome to the AI Agent System</h1>" in response.text

def test_configure():
    response = client.post(
        "/configure",
        data={
            "orchestrator_model": "claude-v1.3",
            "refiner_model": "gpt-3.5-turbo",
            "subagent_model": "gpt-3.5-turbo",
            "goal": "Test Goal",
            "orchestration_strategy": "Strategy 1",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"message": "AI Agent System configured successfully"}

def test_run():
    client.post(
        "/configure", 
        data={
            "orchestrator_model": "claude-v1.3",
            "refiner_model": "gpt-3.5-turbo", 
            "subagent_model": "gpt-3.5-turbo",
            "goal": "Test Goal",
            "orchestration_strategy": "Strategy 1",
        },
    )
    response = client.post("/run")
    assert response.status_code == 200
    assert "result" in response.json()
    assert "orchestrator_model" in response.json()
    assert "refiner_model" in response.json()
    assert "subagent_model" in response.json()
    assert "goal" in response.json()
    assert "orchestration_strategy" in response.json()