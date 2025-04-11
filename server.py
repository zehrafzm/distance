from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod use your domain here
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
                return 0.0  # fallback

        d1 = safe_float(data.get("distance1"))
        d2 = safe_float(data.get("distance2"))
        d3 = safe_float(data.get("distance3"))

        print(f"🔢 Distances received: {d1}, {d2}, {d3}")

        # Coordinates for the sensor points (left, middle, right)
        sensor_x = np.array([0, 0.5, 1])
        sensor_y = np.array([0.5, 0.5, 0.5])  # center row
        sensor_vals = np.array([d1, d2, d3])

        # Create a grid to interpolate over
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, 300),  # width
            np.linspace(0, 1, 100)   # height
        )

        # Interpolate values using linear interpolation
        grid_z = griddata(
            points=(sensor_x, sensor_y),
            values=sensor_vals,
            xi=(grid_x, grid_y),
            method='cubic',
            fill_value=0
        )

        # Normalize to [0, 1] range
        norm_image = np.clip(grid_z / 60.0, 0, 1)

        # Apply colormap
        colormap = plt.get_cmap('plasma')
        colored_img = colormap(norm_image)
        img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("🔥 ERROR IN /heatmap/:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
