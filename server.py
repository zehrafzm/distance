from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/heatmap/")
async def generate_heatmap(request: Request):
    try:
        data = await request.json()

        def safe_float(val):
            try:
                return float(val)
            except:
                return 0.0

        d1 = safe_float(data.get("distance1"))
        d2 = safe_float(data.get("distance2"))
        d3 = safe_float(data.get("distance3"))

        print(f"🔢 Distances received: {d1}, {d2}, {d3}")

        width, height = 300, 150
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

        # Gaussian function generator
        def gaussian(x, y, center_x, center_y, intensity, sigma=0.15):
            return intensity * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

        # Add 3 sensor "heat sources"
        heatmap = (
            gaussian(grid_x, grid_y, 0.0, 0.5, d1) +
            gaussian(grid_x, grid_y, 0.5, 0.5, d2) +
            gaussian(grid_x, grid_y, 1.0, 0.5, d3)
        )

        # Normalize
        norm_heatmap = np.clip(heatmap / 60.0, 0, 1)

        # Apply colormap
        colormap = plt.get_cmap('plasma')
        colored_img = colormap(norm_heatmap)
        img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("🔥 ERROR IN /heatmap/:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
