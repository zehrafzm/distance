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
        def safe_float(val):
            try:    return float(val)
            except: return 0.0

        # Pull out nine distances
        d = [safe_float(data.get(f"distance{i}")) for i in range(1, 10)]
        print(f"📡 Distances received: {d}")

        # Skip if all-zero
        if all(v == 0.0 for v in d):
            return Response(status_code=204)

        # Make a 3×3 array
        sensor_grid = np.array(d).reshape((3, 3))

        # Normalize to [0,1]
        minv, maxv = sensor_grid.min(), sensor_grid.max()
        norm_grid = (sensor_grid - minv) / ( (maxv - minv) + 1e-6 )

        # Create a 300×200 PNG with 3×3 blocks
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        ax.imshow(
            norm_grid,
            cmap="plasma",
            interpolation="nearest",
            origin="lower",
            extent=[0, 1, 0, 1]
        )
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

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
