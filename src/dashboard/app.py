from fastapi import FastAPI, Depends, Request, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import shutil

from src.dashboard.camera_manager import CameraManager

app = FastAPI(title="Urban Detection Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize camera manager once
# Note: Ensure that output.mp4 exists in the project root, or default to 0.
video_source = "output.mp4" if os.path.exists("output.mp4") else 0
manager = CameraManager(source=video_source)
manager.start()

# Mount frontend files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ConfigUpdate(BaseModel):
    conf_threshold: float

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(manager.get_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
async def stats():
    return manager.get_stats()

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    manager.set_conf_threshold(config.conf_threshold)
    return {"status": "success", "conf_threshold": config.conf_threshold}

@app.post("/api/source")
async def update_source(data: dict):
    source = data.get("source", 0)
    manager.change_source(source)
    return {"status": "success", "source": source}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded file
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Automatically switch source to the uploaded file
    manager.change_source(file_path)
    return {"status": "success", "source": file_path, "filename": file.filename}

@app.on_event("shutdown")
def shutdown_event():
    manager.stop()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
