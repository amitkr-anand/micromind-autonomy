import bpy
import math
import os

def latlon_to_mercator(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    return x, y

origin_lat = 31.104
origin_lon = 77.173
ox, oy = latlon_to_mercator(origin_lat, origin_lon)

# Altitude for ~5km footprint at 60deg FOV

CAMERA_ALT_M = 12000

OUTPUT_DIR = "/home/mmuser/micromind/repos/micromind-autonomy/data/synthetic_imagery/shimla_corridor/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Revised waypoints — staying within tile bounds
# Tile: X -52003 to +52003, Y -29259 to +29259
# Moving east from Shimla keeps us within bounds
# All points verified inside tile
#waypoints = [
#   (0,  31.104, 77.173),  # Shimla city
#    (5,  31.120, 77.220),  # East ridge
#    (10, 31.136, 77.267),  # Forest zone
#    (15, 31.152, 77.314),  # Upper ridge
#    (20, 31.168, 77.361),  # Valley approach
#    (25, 31.184, 77.408),  # Ridge crossing
#    (30, 31.200, 77.455),  # Mixed terrain
#    (35, 31.216, 77.502),  # East corridor
#    (40, 31.232, 77.549),  # Valley floor
#    (45, 31.248, 77.596),  # Rampur approach
#    (50, 31.264, 77.630),  # River valley
#    (55, 31.268, 77.585),  # Corridor end
#]

waypoints = [
    (0,  31.260, 76.790),   # shifted east slightly
    (5,  31.242, 76.860),
    (10, 31.224, 76.940),
    (15, 31.206, 77.020),
    (20, 31.188, 77.100),
    (25, 31.170, 77.180),
    (30, 31.152, 77.260),
    (35, 31.134, 77.340),
    (40, 31.116, 77.420),
    (45, 31.098, 77.500),
    (50, 31.080, 77.555),
    (55, 31.062, 77.565),   # shifted west slightly
]


# Verify all in bounds before render
print("Verifying waypoints...")
all_ok = True
for km, lat, lon in waypoints:
    x, y = latlon_to_mercator(lat, lon)
    bx = x - ox
    by = y - oy
    in_x = -52003 < bx < 52003
    in_y = -29259 < by < 29259
    ok = "OK" if (in_x and in_y) else "OUT"
    print(f"  km={km:3d} bx={bx:7.0f} by={by:7.0f} {ok}")
    if ok == "OUT":
        all_ok = False

if not all_ok:
    print("ERROR: Some waypoints outside tile bounds. Aborting.")
    raise SystemExit("Fix waypoints first")

print("All waypoints in bounds. Starting render.")
print()

# Render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.cycles.device = 'GPU'
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    print("CUDA GPU enabled")
except Exception as e:
    print(f"GPU note: {e} — using CPU")
    bpy.context.scene.cycles.device = 'CPU'

bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 640
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGB'

cam = bpy.data.objects.get('Camera.002') or bpy.data.objects.get('Camera')
if cam is None:
    raise RuntimeError(f"Camera not found. Objects: {list(bpy.data.objects.keys())}")
print(f"Camera: {cam.name}")

# Nadir orientation — pointing straight down
cam.rotation_euler = (0, 0, 0)

print(f"Camera Z: {CAMERA_ALT_M}m")
print(f"Expected footprint: ~5km x 5km")
print()

for km, lat, lon in waypoints:
    x, y = latlon_to_mercator(lat, lon)
    bx = x - ox
    by = y - oy

    cam.location.x = bx
    cam.location.y = by
    cam.location.z = CAMERA_ALT_M

    bpy.context.view_layer.update()

    filename = f"frame_km{km:03d}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    bpy.context.scene.render.filepath = filepath

    print(f"Rendering km={km:3d} ({lat:.3f}N {lon:.3f}E)")
    bpy.ops.render.render(write_still=True)

    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  Saved: {filename} ({size//1024}KB)")
    else:
        print(f"  WARNING: {filename} missing")

print()
print("Complete.")
print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}")
