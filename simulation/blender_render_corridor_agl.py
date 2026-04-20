import bpy
import math
import os

# ── AGL mode ──────────────────────────────────────────────────────────────────
# Camera is placed DESIRED_AGL_M above the local terrain surface at every
# waypoint instead of a fixed scene-Z altitude.
DESIRED_AGL_M = 150

# ── Test mode ─────────────────────────────────────────────────────────────────
# Set True to run only km=0 and km=55 without writing rendered images.
TEST_MODE = False

# ── Coordinate helpers ────────────────────────────────────────────────────────
def latlon_to_mercator(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


origin_lat = 31.104
origin_lon = 77.173
ox, oy = latlon_to_mercator(origin_lat, origin_lon)

# ── Terrain height via ray-cast ───────────────────────────────────────────────
# Ray is fired from a high altitude straight down and the hit Z is returned.

def get_terrain_height(terrain_obj, x, y):
    """Return the Z coordinate of the terrain mesh directly below (x, y).

    Uses Blender's evaluated ray_cast() so modifiers (subdivision, displacement)
    are fully applied before the intersection test.

    Returns 0.0 if no hit is found (flat fallback so the camera still moves).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = terrain_obj.evaluated_get(depsgraph)

    ray_origin    = (x, y, 100000.0)   # high above any likely terrain
    ray_direction = (0.0, 0.0, -1.0)   # straight down

    # ray_cast() on a Blender mesh object: returns (hit, location, normal, face_index)
    # NOTE: coordinates are in *object* space, so the terrain object must have no
    # rotation/scale applied, or we convert.  The Shimla DEM is expected to be
    # placed at the world origin with scale=1.
    hit, location, normal, face_index = obj_eval.ray_cast(
        ray_origin,
        ray_direction
    )

    if hit:
        return location.z
    return 0.0


# ── Resolve terrain mesh ───────────────────────────────────────────────────────
TERRAIN_CANDIDATE_NAMES = ["terrain_shimla", "DEM", "Plane", "Terrain", "Terrain.001"]

terrain = None
for name in TERRAIN_CANDIDATE_NAMES:
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == 'MESH':
        terrain = obj
        break

if terrain is None:
    all_objs = [o.name for o in bpy.data.objects if o.type == 'MESH']
    raise RuntimeError(
        f"No terrain mesh found (tried {TERRAIN_CANDIDATE_NAMES}). "
        f"Mesh objects in scene: {all_objs}"
    )

print(f"Terrain object: {terrain.name!r}")

# ── Waypoints ─────────────────────────────────────────────────────────────────
all_waypoints = [
    (0,  31.260, 76.790),
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
    (55, 31.062, 77.565),
]

if TEST_MODE:
    waypoints = [(km, lat, lon) for km, lat, lon in all_waypoints if km in (0, 55)]
    print("TEST_MODE=True — processing km=0 and km=55 only; rendering disabled.")
else:
    waypoints = all_waypoints

# ── Verify waypoints inside tile bounds ───────────────────────────────────────
# Tile: X -52003 to +52003, Y -29259 to +29259
print("\nVerifying waypoints...")
all_ok = True
for km, lat, lon in waypoints:
    x, y = latlon_to_mercator(lat, lon)
    bx = x - ox
    by = y - oy
    in_x = -52003 < bx < 52003
    in_y = -29259 < by < 29259
    ok = "OK" if (in_x and in_y) else "OUT"
    print(f"  km={km:3d}  bx={bx:8.0f}  by={by:8.0f}  {ok}")
    if ok == "OUT":
        all_ok = False

if not all_ok:
    raise SystemExit("ERROR: Some waypoints outside tile bounds. Fix waypoints first.")

print("All waypoints in bounds.\n")

# ── Render settings (set even in TEST_MODE so scene is consistent) ─────────────
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

# ── Camera ─────────────────────────────────────────────────────────────────────
cam = bpy.data.objects.get('Camera.002') or bpy.data.objects.get('Camera')
if cam is None:
    raise RuntimeError(f"Camera not found. Objects: {list(bpy.data.objects.keys())}")
print(f"Camera: {cam.name}")

# Nadir orientation — pointing straight down
cam.rotation_euler = (0, 0, 0)

# Footprint at 150 m AGL, 60° FOV (horizontal = vertical for square sensor):
#   half-angle = 30°  →  footprint_half = 150 * tan(30°)
#   full footprint = 2 * 150 * tan(30°)  ≈ 173 m
footprint_m = 2 * DESIRED_AGL_M * math.tan(math.radians(60 / 2))
print(f"AGL: {DESIRED_AGL_M} m  |  FOV: 60°  |  Expected footprint: {footprint_m:.1f} m x {footprint_m:.1f} m")
print()

# ── Per-waypoint loop ──────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/mmuser/micromind/repos/micromind-autonomy/data/synthetic_imagery/shimla_corridor/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for km, lat, lon in waypoints:
    x, y = latlon_to_mercator(lat, lon)
    bx = x - ox
    by = y - oy

    # Terrain height directly below this waypoint
    terrain_z = get_terrain_height(terrain, bx, by)

    # Set camera to exactly DESIRED_AGL_M above local ground
    cam.location.x = bx
    cam.location.y = by
    cam.location.z = terrain_z + DESIRED_AGL_M

    bpy.context.view_layer.update()

    # Per-frame diagnostic
    print(
        f"km={km:3d}  "
        f"bx={bx:8.0f}  by={by:8.0f}  "
        f"terrain_z={terrain_z:8.1f} m  "
        f"camera_z={cam.location.z:8.1f} m  "
        f"agl={DESIRED_AGL_M:.1f} m  "
        f"footprint={footprint_m:.1f} m"
    )

    if TEST_MODE:
        # Skip actual render in test mode — position diagnostics are sufficient
        continue

    filename = f"frame_km{km:03d}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    bpy.context.scene.render.filepath = filepath

    bpy.ops.render.render(write_still=True)

    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  Saved: {filename} ({size // 1024} KB)")
    else:
        print(f"  WARNING: {filename} missing")

print()
if TEST_MODE:
    print("Test run complete (no images written).")
else:
    print("Render complete.")
    print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}")
