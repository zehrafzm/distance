from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
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

latest_image_bytes = None

@app.post("/heatmap/")
async def generate_heatmap(request: Request):
    global latest_image_bytes
    try:
        data = await request.json()
        def safe_float(v):
            try:    return float(v)
            except: return 0.0

        # 1) Read nine distances into a flat list
        d = [safe_float(data.get(f"distance{i}")) for i in range(1, 10)]
        print(f"📡 Distances received: {d}")

        # 2) If all zero, skip
        if all(val == 0.0 for val in d):
            return Response(status_code=204)

        # 3) Build a 3×3 array
        sensor_grid = np.array(d).reshape((3, 3))

        # 4) Normalize to [0,1]
        mn, mx = sensor_grid.min(), sensor_grid.max()
        norm_grid = (sensor_grid - mn) / ( (mx - mn) + 1e-6 )

        # 5) Map through plasma colormap → shape (3,3,4)
        cmap = plt.get_cmap("plasma")
        colored_rgba = cmap(norm_grid)

        # 6) Drop alpha, convert to uint8 (3×3×3)
        colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)

        # 7) Make a tiny PIL image (3×3 pixels)
        tiny = Image.fromarray(colored_rgb, mode="RGB")

        # 8) Upscale to e.g. 300×300 with NEAREST so each block is exactly 100×100px
        img = tiny.resize((300, 300), resample=Image.NEAREST)

        # 9) Write PNG to memory
        buf = BytesIO()
        img.save(buf, format="PNG")
        latest_image_bytes = buf.getvalue()

        print("✅ Block heatmap updated.")
        return Response(status_code=200)

    except Exception as e:
        print("🔥 Error in /heatmap/:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/image")
async def get_latest_image():
    global latest_image_bytes
    if latest_image_bytes:
        return Response(content=latest_image_bytes, media_type="image/png")
    return Response(status_code=204)
