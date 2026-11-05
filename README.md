# Air Canvas

Draw in the air with your index finger. MediaPipe hand tracking turns a webcam
into a paint surface: hover to see the cursor, pinch thumb + index to commit a
stroke.

This was the first computer-vision build (year 2 coursework) and the origin
project that later grew into [Gesture-Control](https://github.com/flogduse/gesture-control),
which replaces the ad-hoc pinch check with a tested, hysteresis-based gesture
engine.

## What it does

- Live webcam video with a hand landmark skeleton overlay.
- A toolbar across the top: 6 colours (blue, green, red, yellow, magenta, white
  eraser) and 4 brush sizes — click a swatch by moving the hand over it and
  pinching.
- Pinch distance is the only gesture, tuned by pixel distance (< 40 px at 720p).
- The canvas accumulates strokes so the drawing persists over the mirrored feed.
- `Q` quits.

## Run it

```powershell
pip install opencv-python numpy mediapipe
python air_canvas.py
```

The MediaPipe hand-landmarker model file (`hand_landmarker.task`) is included in
this repository, so nothing downloads at first run.

## Honest notes on the design

- **The pinch threshold is a raw pixel distance**, not a palm-relative ratio.
  Move the hand away from the camera and the same physical pinch changes size,
  so clicks misfire at distance. Recognising exactly this problem is what led to
  the palm-scale normalisation in the gesture engine built on top of it.
- **Palm-relative ratios** (see `gesture-control`) hold their meaning at 30 cm
  and 1.5 m; this project pre-dates that fix.
