from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://distance-web.vercel.app"]
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

        x = [0, 1, 2]
        y = [d1, d2, d3]

        x_interp = np.linspace(0, 2, 300)
        y_interp = np.interp(x_interp, x, y)
        image = np.tile(y_interp, (100, 1))
        norm_image = np.clip(image / 60.0, 0, 1)
        colormap = plt.get_cmap('plasma')
        colored_img = colormap(norm_image)
        img = Image.fromarray((colored_img[:, :, :3] * 255).astype(np.uint8))

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("🔥 ERROR IN /heatmap/:", e)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})
