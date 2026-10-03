# Real-Time Hand Sign Recognition Using Landmark Extraction

This Python project recognizes six static hand signs from webcam input using OpenCV, MediaPipe hand landmarks, and a trained PyTorch neural network. It displays gesture labels and confidence scores, and can trigger a shadow clone visual effect using person segmentation. The dataset is not provided but there is provision to retrain the model on new hand signs as well.

## Pipeline

Webcam input -> MediaPipe Hands -> 126 landmark features -> saved StandardScaler -> PyTorch MLP -> gesture label and confidence -> optional visual effect

## Supported Gestures

- Horse (`horse`)
- OK (`ok`)
- Ram (`ram`)
- Shadow Clone (`shadow_clone`)
- Snake (`snake`)
- Thumbs Up (`thumbs_up`)

## Current Status

The core prototype is implemented through Phase 7: webcam hand tracking, labeled sample collection, model training and evaluation, live inference, and gesture-triggered visual effects. Trained model weights, feature scaler, label map, and model configuration are included in `models/`.

The training notebook records 1,206 samples (201 per class), an 80/20 stratified train/test split, and 100% test accuracy on 242 held-out samples. These are recorded notebook results from a random sample split; performance on new users or recording sessions has not been established by that evaluation.

## Main Files

| File | Purpose |
| --- | --- |
| `src/webcam_test.py` | Captures webcam frames and displays hand landmarks. |
| `src/collect_landmarks.py` | Records labeled samples with up to two hands into a CSV file. |
| `src/utils.py` | Extracts 126 features, orders hands by handedness, and fills missing hands with zeros. |
| `src/model.py` | Defines the MLP: 126 inputs, hidden layers of 128 and 64 units, and six output classes. |
| `notebooks/01_train_gesture_classifier.ipynb` | Preprocesses data, trains and evaluates the model, and saves inference artifacts. |
| `src/webcam_inference.py` | Loads saved artifacts and displays live predictions with a confidence threshold. |
| `src/effects.py` | Creates faded, shifted copies of the segmented person for the shadow clone effect. |
| `requirements.txt` | Lists dependencies, including MediaPipe pinned to version 0.10.14. |

## Setup Instructions

```powershell
git clone https://github.com/kevinsukesh2/Hand_sign_recognition.git
cd Hand_sign_recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A webcam is required for capture and live inference. Run the following commands from the repository root.

## Run Live Recognition

```powershell
python src/webcam_inference.py
```

The default confidence threshold is 0.7. Predictions below the threshold display as `Unknown`. A confident `shadow_clone` prediction activates the effect when MediaPipe Selfie Segmentation is available. Press `q` to quit.

To change the threshold or disable the effect:

```powershell
python src/webcam_inference.py --threshold 0.8 --disable-shadow-clone-effect
```

To test hand tracking alone:

```powershell
python src/webcam_test.py
```

## Collect Data and Retrain

The original landmark CSV is excluded from Git. The saved model can be used without that dataset; retraining requires collecting new samples or supplying a compatible CSV.

```powershell
python src/collect_landmarks.py --label horse --duration 100 --interval 0.5
```

Press `s` to start collection after a countdown, or `q` to quit. Samples are saved to `data/gestures_landmarks.csv`. Repeat for each supported gesture using its exact label above.

Launch Jupyter from the repository root, open `notebooks/01_train_gesture_classifier.ipynb`, and run its cells to train and evaluate the classifier. Training uses feature standardization, cross-entropy loss, Adam, and 100 epochs, and writes the model artifacts to `models/`.

```powershell
jupyter notebook
```

## Future Work

- Evaluate recognition across new users, recording sessions, and lighting conditions.
- Add a demo video and broader reliability testing.
- Extend static gesture recognition to gesture sequences, such as Fireball Jutsu.
