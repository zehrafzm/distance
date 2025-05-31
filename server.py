from fastapi import FastAPI, Request, Response, JSONResponse
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

        # Pull out eight distances (you said you have eight sensors right now)
        d = [safe_float(data.get(f"distance{i}")) for i in range(1, 9)]
        print(f"📡 Distances received: {d}")

        # If all eight are zero, skip
        if all(v == 0.0 for v in d):
            print("⚠️ Skipped frame due to all-zero values")
            return Response(status_code=204)

        # 1) Define your eight sensor (x,y) positions in [0,1]×[0,1].
        #    You can tweak these coordinates to match your physical layout.
        sensor_x = np.array([0.2, 0.4, 0.6, 0.8,
                             0.2, 0.4, 0.6, 0.8])
        sensor_y = np.array([0.4, 0.4, 0.4, 0.4,
                             0.8, 0.8, 0.8, 0.8])
        sensor_vals = np.array(d)
        points = np.column_stack((sensor_x, sensor_y))  # shape (8,2)

        # 2) Build a fine grid (e.g. 200×300) over the unit square:
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, 300),
            np.linspace(0, 1, 200)
        )
        flat_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # 3) Interpolate via RBF so we get a “height” (distance) at each grid point.
        #    smoothing=0.5 or similar usually works. You can experiment.
        rbf = RBFInterpolator(points, sensor_vals, smoothing=0.5)
        grid_z = rbf(flat_grid).reshape(grid_x.shape)  # shape (200,300)

        # 4) Now choose how you want to color “height” → “color”. 
        #    Let’s say your sensor distances range roughly 0–140 cm.
        #    We’ll create a discrete colormap with four bands:
        #      - blue for 0–40 cm
        #      - green for 40–80 cm
        #      - yellow for 80–120 cm
        #      - reddish‐brown for 120–140+ cm
        #
        #    First, normalize grid_z into [0,1] by dividing by 140.0 (clipping >1):
        norm = grid_z / 140.0
        norm = np.clip(norm, 0.0, 1.0)

        #    Then build a custom colormap with four distinct colors and exact cutpoints.
        from matplotlib.colors import BoundaryNorm, ListedColormap

        #  Define the RGBA tuples for each band (in matplotlib’s “plasma” style or custom).
        #  You can tweak these hex‐codes or RGB triples as you like:
        colors = [
            "#2c115f",  # dark blue  (for the lowest band: 0–0.286)
            "#1f3284",  # medium‐blue/green (0.286–0.571)
            "#66b32e",  # green (0.571–0.857)
            "#fdae61",  # yellow‐orange (0.857–1.0)
            "#8b0000"   # dark red for anything above 1.0  
        ]
        #    Because we have 4 bands, we define 5 boundaries in “height‐normalized” space:
        bounds = [0.0, 0.286, 0.571, 0.857, 1.0]
        cmap = ListedColormap(colors)
        norm_map = BoundaryNorm(bounds, cmap.N)  # 4 “intervals”, index 0→4

        # 5) Now create a figure, draw a pcolormesh (or imshow) using that colormap.
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        mesh = ax.pcolormesh(
            grid_x, grid_y, grid_z / 140.0,  # we still pass the *values* in [0,1]
            cmap=cmap, norm=norm_map,
            shading="auto"
        )

        # 6) Overlay contour lines (in black) at your choice of actual distance‐values.
        #    Let’s draw lines every 20 cm: 20, 40, 60, 80, 100, 120.
        contour_levels = [20/140.0, 40/140.0, 60/140.0, 80/140.0, 100/140.0, 120/140.0]
        cs = ax.contour(
            grid_x, grid_y, grid_z/140.0, levels=contour_levels, colors="k", linewidths=0.8
        )
        ax.clabel(cs, fmt=lambda x: f"{int(x*140)} cm", inline=True, fontsize=6)

        ax.axis("off")
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)

        # 7) Dump into a PNG in memory
        buf = BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        latest_image_bytes = buf.getvalue()
        print("✅ Heatmap with contour‐lines updated.")
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
