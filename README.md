# People Counter with YOLOv8 and SORT

A real-time people counting system that uses YOLOv8n for person detection and SORT (Simple Online Realtime Tracking) for object tracking. The system can count people moving up and down across a defined line in a video stream.

## Features

- Real-time person detection using YOLOv8n
- Accurate person tracking with the SORT algorithm
- Bi-directional counting (upward and downward movement)
- Visual feedback with bounding boxes and tracking IDs
- Customizable detection confidence threshold
- Live count display on video feed

## Credits

This project uses the SORT (Simple Online and Realtime Tracking) algorithm:
- **SORT**: Copyright (C) 2016-2020 Alex Bewley [GitHub Repository](https://github.com/abewley/sort)
- **YOLOv8**: Created by Ultralytics

## How It Works

1. **Detection**: YOLOv8n detects people in each frame
2. **Tracking**: SORT algorithm assigns and maintains unique IDs for each person
3. **Counting**: The system counts people when they cross a predefined line:
   - Tracks movement direction (up/down)
   - Updates counters based on crossing direction
   - Provides visual feedback with color-coded lines and boxes
