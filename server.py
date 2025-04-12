from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
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

        # ✅ Proper indentation here
        if d1 == 0.0 and d2 == 0.0 and d3 == 0.0:
            print("❌ Skipping zeroed-out frame")
            return Response(status_code=204)

        sensor_x = np.array([0.25, 0.5, 0.75])
        sensor_y = np.array([0.3, 0.7, 0.4])
        sensor_vals = np.array([d1, d2, d3])

        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, 300),
            np.linspace(0, 1, 200)
        )

        points = np.column_stack((sensor_x, sensor_y))
        rbf = RBFInterpolator(points, sensor_vals, smoothing=5.0)
        
        flat_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        grid_z = rbf(flat_grid).reshape(grid_x.shape)


        norm_image = np.clip(grid_z / 60.0, 0, 1)
        colormap = plt.get_cmap('plasma')
        colored_img = colormap(norm_image)
        img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))

        buf = BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        print("🔥 ERROR IN /heatmap/:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
