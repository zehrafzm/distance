# server.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import mimetypes
from conversion import process_file, remux_video

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://192.168.1.13:3000"] for stricter access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory
OUTPUT_DIR = "processed_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/process/")
async def process_file_endpoint(
    file: UploadFile = File(...),
    lower_threshold: int = Form(...),
    upper_threshold: int = Form(...),
    is_black_background: bool = Form(...),
    high_quality: bool = Form(False)  
):
    # Save uploaded file
    input_path = os.path.join(OUTPUT_DIR, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Determine output paths
    ext = os.path.splitext(file.filename)[-1].lower()
    processed_path = os.path.join(OUTPUT_DIR, f"processed_{file.filename}")
    final_path = processed_path

    # Process file (image or video)
    process_file(
        input_path,
        processed_path,
        lower_threshold,
        upper_threshold,
        is_black_background,
        high_quality
    )

    # If video, remux for browser playback
    if ext in [".mp4", ".avi", ".mov"]:
        final_path = os.path.join(OUTPUT_DIR, f"remuxed_{file.filename}")
        remux_video(processed_path, final_path, input_path) 

    # Detect MIME type
    media_type, _ = mimetypes.guess_type(final_path)
    return FileResponse(final_path, media_type=media_type, filename=os.path.basename(final_path))