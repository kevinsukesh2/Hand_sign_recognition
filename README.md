# Real-Time Hand Sign Recognition Using Landmark Extraction

This project will use a webcam to recognize static hand signs using MediaPipe hand landmarks and a neural network classifier.

## Planned Pipeline

Webcam input -> MediaPipe Hands -> landmark extraction -> MLP classifier -> gesture label/action text

## Planned Gestures

- Snake
- Ram
- Monkey
- Boar
- Horse
- Tiger

## Current Status

Phase 1: project setup only

## Future Phases

- Phase 2: webcam + MediaPipe hand landmark test
- Phase 3: landmark data collection
- Phase 4: collect dataset
- Phase 5: train neural network classifier
- Phase 6: webcam inference
- Phase 7: action/script text output
- Phase 8: testing and cleanup
- Phase 9: report/demo preparation
- Phase 10: future Fireball Jutsu sequence detection

## Setup Instructions

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
