#---------------------------------------------------------------------------------
# File : server.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : Prompts and strategies for AI Agent orchestration
# Purp : The base prompts and iteration strategies for the AI Agent application.
#---------------------------------------------------------------------------------

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from agents import run_model

from orchestrator import ModelConfig, AgentConfig, run_orchestrator_loop

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



agent_config: AgentConfig = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/configure")
async def configure(
    orchestrator_model: str = Form(...),
    refiner_model: str = Form(...),
    subagent_model: str = Form(...),
    goal: str = Form(...),
    orchestration_strategy: str = Form(...)
):
    global agent_config
    agent_config = AgentConfig(
        orchestrator_model=orchestrator_model,
        refiner_model=refiner_model,
        subagent_model=subagent_model,
        goal=goal,
        orchestration_strategy=orchestration_strategy
    )
    return JSONResponse(content={"message": "AI Agent System configured successfully"})

@app.post("/run")
async def run():
    try:
        orchestrator_output = await run_model(agent_config.orchestrator_model, agent_config.goal)
        refiner_output = await run_model(agent_config.refiner_model, orchestrator_output)
        subagent_output = await run_model(agent_config.subagent_model, refiner_output)

        if agent_config.orchestration_strategy == "Strategy 1":
            final_output = await orchestrate_strategy1(orchestrator_output, refiner_output, subagent_output)
        elif agent_config.orchestration_strategy == "Strategy 2":
            final_output = await orchestrate_strategy2(orchestrator_output, refiner_output, subagent_output)
        elif agent_config.orchestration_strategy == "Strategy 3":
            final_output = await orchestrate_strategy3(orchestrator_output, refiner_output, subagent_output)
        elif agent_config.orchestration_strategy == "Strategy 4":
            final_output = await orchestrate_strategy4(orchestrator_output, refiner_output, subagent_output)
        else:
            raise ValueError(f"Unsupported orchestration strategy: {agent_config.orchestration_strategy}")

        return JSONResponse(content={
            "result": final_output,
            "orchestrator_model": agent_config.orchestrator_model,
            "refiner_model": agent_config.refiner_model,
            "subagent_model": agent_config.subagent_model,
            "goal": agent_config.goal,
            "orchestration_strategy": agent_config.orchestration_strategy
        })
    except Exception as e:
        error_message = f"An error occurred during execution: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)