from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

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
            try:
                return float(val)
            except:
                return 0.0

        # Pull out six distances
        d = [safe_float(data.get(f"distance{i}")) for i in range(1, 10)]
        print(f"📡 Distances received: {d}")

        # If all six are zero, skip
        if all(v == 0.0 for v in d):
            print("⚠️ Skipped frame due to all-zero values")
            return Response(status_code=204)

        # Define your six sensor locations
        sensor_x = np.array([0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8])
        sensor_y = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8])

        sensor_vals = np.array(d)
        points = np.column_stack((sensor_x, sensor_y))

        # Build the grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, 300),
            np.linspace(0, 1, 200)
        )
        flat_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # RBF interpolation
        rbf = RBFInterpolator(points, sensor_vals, smoothing=1.0)
        grid_z = rbf(flat_grid).reshape(grid_x.shape)

        # Normalize and colorize
        # Normalize and colorize (using numpy’s standalone ptp)
        norm = (grid_z - grid_z.min()) / (np.ptp(grid_z) + 1e-6)

        cmap = plt.get_cmap("plasma")
        colored = cmap(norm)
        img = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))

        # Write to in-memory PNG
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
