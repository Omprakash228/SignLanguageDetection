# Sign Language Detection

A real-time sign language detection system that recognizes three sign language gestures: "hello", "thanks", and "peace". The project uses MediaPipe Holistic for pose, face, and hand landmark detection, and an LSTM neural network for sequence classification.

## Features

- Real-time sign language gesture recognition using webcam
- Detects three gestures: hello, thanks, peace
- Uses MediaPipe Holistic for extracting body, face, and hand keypoints
- LSTM-based neural network for temporal sequence classification
- Visual feedback with probability visualization

## Demo

Watch the demo video to see the sign language detection system in action:

https://github.com/user-attachments/assets/5456c7ed-13b1-40d7-81b2-39d51ab68a6c

## Technologies Used

- **MediaPipe**: For holistic pose, face, and hand landmark detection
- **TensorFlow/Keras**: For building and training the LSTM model
- **OpenCV**: For video capture and image processing
- **NumPy**: For data manipulation
- **scikit-learn**: For data preprocessing and evaluation

## Project Structure

```
SignLanguageDetection/
├── SignLanguageDetector.ipynb    # Main Jupyter notebook with all code
├── MP_Data/                      # Training data (numpy arrays)
│   ├── hello/                    # Hello gesture sequences
│   ├── thanks/                   # Thanks gesture sequences
│   └── peace/                 # Peace gesture sequences
├── Logs/                         # TensorBoard training logs
└── README.md                     # This file
```

## Installation

1. Install the required dependencies:

```bash
pip install tensorflow==2.5.0 tensorflow-gpu==2.5.0 opencv-python mediapipe sklearn matplotlib
```

Note: If you don't have a GPU, you can install `tensorflow` instead of `tensorflow-gpu`.

## Usage

### Training the Model

1. Open `SignLanguageDetector.ipynb` in Jupyter Notebook
2. Run the cells to:
   - Set up MediaPipe Holistic model
   - Collect training data (30 sequences per gesture, 30 frames per sequence)
   - Preprocess the data
   - Train the LSTM model
   - Save the model as `action.h5`

### Real-time Detection

Run the real-time detection cell in the notebook. The system will:
- Capture video from your webcam
- Extract keypoints using MediaPipe
- Make predictions using the trained LSTM model
- Display the detected gesture with probability visualization
- Press 'q' to quit

## Model Architecture

The LSTM model consists of:
- 3 LSTM layers (64, 128, 64 units)
- 2 Dense layers (64, 32 units)
- Output layer with softmax activation for 3 classes
- Total parameters: ~596,675

## Data Collection

The training data consists of:
- 3 gestures: hello, thanks, peace
- 30 sequences per gesture
- 30 frames per sequence
- Each frame contains 1662 keypoints (pose: 132, face: 1404, left hand: 63, right hand: 63)
