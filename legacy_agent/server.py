# --------------------------------------------------------------------------------
# File : server.py
# Auth : Dan Gilbert
# Date : 5/27/2024
# Desc : Prompts and strategies for AI Agent orchestration
# Purp : The base prompts and iteration strategies for the 
#        AI Agent application.
# --------------------------------------------------------------------------------

import os
import time
from starlette.responses import HTMLResponse
from starlette.responses import RedirectResponse
from starlette.responses import StreamingResponse

from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import motor.motor_asyncio

from orchestrator import ModelConfig, AgentConfig, run_orchestrator_loop
from sse_starlette.sse import EventSourceResponse
from sh import tail

from rich.console import Console
#from threading import Thread
import multiprocessing
from fastapi.middleware.cors import CORSMiddleware

##############################################################################
# Create App and static/templates
##############################################################################


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################################################################
# Connect to Mongo DB
##############################################################################


MONGO_CONN = os.environ['MONGO_CONN']
MONGO_PORT = int(os.environ['MONGO_PORT'])
MONGO_DBNAME = os.environ['MONGO_DBNAME']

client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_CONN, MONGO_PORT, tls=True,
            tlsAllowInvalidCertificates=True)
mongo_db = client[MONGO_DBNAME]


##############################################################################
# Home Landingg Page
##############################################################################


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


##############################################################################
# Save/Update Agent Into Mongo DB
##############################################################################


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
    print(f"save_agent: {name}")
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


##############################################################################
# View Agent by ID
##############################################################################


@app.get("/view_agent/{id}/", response_class=HTMLResponse)
async def view_agent(id: str, request: Request):
    print(f"view_agent: {id}")
    cfg = await mongo_db[MONGO_DBNAME].find_one({"_id": id})

    context = {"request": request,
               "agent": cfg,
               "layout": "all"}

    return templates.TemplateResponse("view_agent.html", context)


##############################################################################
# Logfile streamer, scroll_tick needs to be > 0.1s to avoid overflow/hang
##############################################################################

SCROLL_TICK = 0.1
proc_for_loop = {}

async def logfile_reader(request: Request, id: str):
    logfile = f"logs/run_orch_loop_{id}.log"
    print(f"logfile_reader: {logfile}")
    for line in tail("-f", logfile, _iter=True):
        if await request.is_disconnected():
            print("client disconnected!!!")
            global proc_for_loop
            if id in proc_for_loop:
                print(f"Terminating process!")
                proc_for_loop[id].terminate()
                del proc_for_loop[id]
                print(f"Done :)")
            break
        yield line
        time.sleep(SCROLL_TICK)


@app.get("/stream_loop_logs/{id}/", response_class=EventSourceResponse)
async def stream_loop_logs(id: str, request: Request):
    print(f"stream_loop_logs: {id}")
    filepath = f"logs/run_orch_loop_{id}.log"
    if os.path.exists(filepath):
        event_generator = logfile_reader(request=request, id=id)
        return EventSourceResponse(event_generator)
    print(f"stream_loop_logs: opps file doesn't exist yet {filepath}")
    return ''


##############################################################################
# Logfile streamer, scroll_tick needs to be > 0.1s to avoid overflow/hang
##############################################################################



@app.get("/run_orch_loop/{id}/", response_class=HTMLResponse)
async def run_orch_loop(id: str, request: Request):
    print(f"run_orch_loop: Getting config from DB {id}")
    cfg = await mongo_db[MONGO_DBNAME].find_one({"_id": id})
    model = ModelConfig(**cfg['model'])
    cfg_vals = {k: v for k, v in cfg.items() if k != 'model'}
    agent = AgentConfig(model=model, **cfg_vals)
    filepath = f"logs/run_orch_loop_{id}.log"
    print(f"run_orch_loop: Starting console with filepath {filepath}")
    console = Console(file=open(filepath, "wt"), record=True, width=80)
    print(f"run_orch_loop: Launching thread for config {id}")
    proc = multiprocessing.Process(target=run_orchestrator_loop, args=(agent, console))
    #thread = Thread(target=run_orchestrator_loop, args=(agent, console))
    print(f"run_orch_loop: Starting Orchestrator loop with thread, logfile={filepath}")
    proc.start()
    global proc_for_loop
    proc_for_loop[id] = proc
    context = {"request": request,
               "agent": cfg,
               "layout": "all"}
    return templates.TemplateResponse("view_agent.html", context)



@app.get("/download_project/{id}/", response_class=StreamingResponse)
async def download_project(id: str, request: Request):

    filename = f"{id}_final.zip"
    filepath = f"output/{filename}"

    if not os.path.exists(filepath):
        return {"error": f"file {filepath} doesn't exist, did you wait for loop to complete?"}
    
    def iterfile():
        with open(filepath, mode="rb") as file_like:  # 
            yield from file_like  # 

    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"'
    }
    return StreamingResponse(iterfile(), headers=headers)

##############################################################################
# Done:)
##############################################################################
