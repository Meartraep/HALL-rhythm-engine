# HALL Rhythm Engine

A precision rhythm visualization and click-track generation engine for **HALL**.

---

## Overview

HALL Rhythm Engine is a tool designed to:

- Generate high-precision click tracks from complex tempo maps
- Visualize time signatures and beat progression in real time
- Handle irregular meters, tempo ramps, and hybrid BPM structures
- Provide adaptive click gain based on music loudness

This project is primarily intended for **analysis, learning, and non-commercial use**.

---

## Demo Features

- Dynamic BPM mapping (including non-linear tempo changes)
- Time signature-aware beat segmentation
- Real-time polygon-based rhythm visualization
- Adaptive click track (volume auto-adjusts with music)

---

## Requirements

- Python 3.9+
- ffmpeg (must be available in PATH)

Download FFmpeg: https://ffmpeg.org/download.html

---

## Installation

```bash
git clone https://github.com/Meartraep/HALL-rhythm-engine.git
cd HALL-rhythm-engine
pip install -r requirements.txt
```
Open HALL.zip.001 with an unzip software and extract HALL.flac
Then run:

```bash
python HALL.py
```

---

## License

This project is licensed under CC-BY-NC-SA 4.0.

Non-commercial use only
Must provide attribution
Derivatives must use the same license

---

## Credits
Music: Frums
https://frums.bandcamp.com/track/hall
https://frums.xyz
Author: Meartraep
Assisted by AI

---

## Notes
This project does NOT include FFmpeg.
Users must install FFmpeg separately due to licensing requirements.
