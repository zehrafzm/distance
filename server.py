from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

app = FastAPI()

# Allow all origins (or restrict as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global image buffer
latest_image_bytes = None

@app.post("/heatmap/")
async def generate_heatmap(request: Request):
    global latest_image_bytes
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

        print(f"📡 Distances received: {d1}, {d2}, {d3}")

        if d1 == 0.0 and d2 == 0.0 and d3 == 0.0:
            print("⚠️ Skipped frame due to all-zero values")
            return Response(status_code=204)

        sensor_x = np.array([0.25, 0.5, 0.75])
        sensor_y = np.array([0.3, 0.7, 0.4])
        sensor_vals = np.array([d1, d2, d3])
        points = np.column_stack((sensor_x, sensor_y))

        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, 300),
            np.linspace(0, 1, 200)
        )
        flat_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        rbf = RBFInterpolator(points, sensor_vals, smoothing=5.0)
        grid_z = rbf(flat_grid).reshape(grid_x.shape)

        norm_image = (grid_z - np.min(grid_z)) / (np.max(grid_z) - np.min(grid_z) + 1e-6)
        colormap = plt.get_cmap('plasma')
        colored_img = colormap(norm_image)
        img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))

        buf = BytesIO()
        img.save(buf, format="PNG")
        latest_image_bytes = buf.getvalue()

        print("✅ Heatmap updated.")
        return Response(status_code=200)

    except Exception as e:
        print("🔥 Error in /heatmap/:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/image")
async def get_latest_image():
    global latest_image_bytes
    if latest_image_bytes:
        return Response(content=latest_image_bytes, media_type="image/png")
    else:
        return Response(status_code=204)
