# --------------------------------------------------------------------------------
# File : server.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : Prompts and strategies for AI Agent orchestration
# Purp : The base prompts and iteration strategies for the 
#        AI Agent application.
# --------------------------------------------------------------------------------

import os

from starlette.responses import HTMLResponse
from starlette.responses import JSONResponse
from starlette.responses import RedirectResponse

from fastapi import FastAPI, Request, Form

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.encoders import jsonable_encoder
import motor.motor_asyncio

from agents import run_model

from orchestrator import ModelConfig, AgentConfig, run_orchestrator_loop

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


MONGO_CONN = os.environ['MONGO_CONN']
MONGO_PORT = int(os.environ['MONGO_PORT'])
MONGO_DBNAME = os.environ['MONGO_DBNAME']

client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_CONN, MONGO_PORT, tls=True, 
            tlsAllowInvalidCertificates=True)
mongo_db = client[MONGO_DBNAME]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/save_agent", response_class=HTMLResponse)
async def save_agent(
    name: str = Form(...),
    orchestrator_model: str = Form(...),
    refiner_model: str = Form(...),
    subagent_model: str = Form(...),
    task_iter: int = Form(...),
    refine_iter: int = Form(...),
    objective: str = Form(...),
    orchestration_strategy: str = Form(...)
):
    model = ModelConfig(orchestrator_model=orchestrator_model,
                        refiner_model=refiner_model,
                        subagent_model=subagent_model,
                        task_iter=task_iter,
                        refine_iter=refine_iter,
                        strategy=orchestration_strategy)
    agent = AgentConfig(name=name, objective=objective, model=model)
    agent = jsonable_encoder(agent)

    new_agent = await mongo_db[MONGO_DBNAME].insert_one(agent)
    find_id = {"_id": new_agent.inserted_id}
    created_agent = await mongo_db[MONGO_DBNAME].find_one(find_id)
    newurl = app.url_path_for('view_agent', id=created_agent["_id"])
    return RedirectResponse(newurl, status_code=303)

    return JSONResponse(content={"message": "AI Agent System saved to DB!"})



@app.get("/view_agent/{id}/", response_class=HTMLResponse)
async def view_agent(id: str, request: Request):
    cfg = await mongo_db[MONGO_DBNAME].find_one({"_id": id})

    context = {"request": request,
               "agent": cfg,
               "layout": "all"}

    return templates.TemplateResponse("view_agent.html", context)


@app.get("/run_orch_loop/{id}/", response_class=HTMLResponse)
async def run_orch_loop(id: str, request: Request):
    cfg = await mongo_db[MONGO_DBNAME].find_one({"_id": id})

    context = {"request": request,
               "agent": cfg,
               "layout": "all"}

    return templates.TemplateResponse("view_agent.html", context)





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