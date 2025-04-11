from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/heatmap/")
async def generate_heatmap(request: Request):
    data = await request.json()
    d1 = data.get("distance1", 0)
    d2 = data.get("distance2", 0)
    d3 = data.get("distance3", 0)

    # Create a 1D distance array and interpolate to create smooth 2D image
    x = [0, 1, 2]
    y = [d1, d2, d3]
    x_interp = np.linspace(0, 2, 300)
    y_interp = np.interp(x_interp, x, y)
    image = np.tile(y_interp, (100, 1))

    # Normalize to colormap (e.g. from 0 to 60 cm)
    norm_image = np.clip(image / 60.0, 0, 1)

    # Convert to RGB image using colormap
    colormap = plt.get_cmap('plasma')
    colored_img = colormap(norm_image)
    img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))

    # Save image to buffer
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return FileResponse(buf, media_type="image/png", filename="heatmap.png")
