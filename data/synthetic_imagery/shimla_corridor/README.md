# Synthetic UAV Imagery — Shimla Corridor

## Source

Blender 3.6+ with BlenderGIS addon.
DEM: Copernicus GLO-30 Shimla tile
Texture: Sentinel-2 L2A true colour
Altitude: 150m AGL
Camera FOV: 60 degrees
Sun: azimuth 135deg, elevation 45deg
Render size: 640 x 640 pixels

## Frame naming

frame_km000.png — Shimla city centre 31.104N 77.173E
frame_km005.png — km 5
frame_km010.png — km 10
frame_km015.png — km 15
frame_km020.png — km 20
frame_km025.png — km 25
frame_km030.png — km 30
frame_km035.png — km 35
frame_km040.png — km 40
frame_km045.png — km 45
frame_km050.png — km 50
frame_km055.png — corridor end

## Purpose

Cross-modal TRN validation.
Query images for phase correlation matching against DEM hillshade
reference tiles. Validates operational TRN performance before HIL.

## Run validation

```
python scripts/validate_cross_modal_trn.py \
  --frames data/synthetic_imagery/shimla_corridor/ \
  --corridor shimla_local
```

## Sensor substitution contract

These frames are the first realisation of the operational camera
pipeline (Blender → future Gazebo → HIL EO camera). The interface
to PhaseCorrelationTRN.match() is identical in all three cases:
numpy uint8 array. No architecture change at HIL.

## Status

Delivered: 15 April 2026 by Programme Director.
12 frames received. Cross-modal validation run in Gate 6.
