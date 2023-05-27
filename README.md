

# Real Time Sign Lanuage Detection

This repository contains code for sign language recognition using deep learning. The landmark positions for face and hand are generated with the help of [mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md) and using these landmarks, a model to detect various sign lanuage gestures is created and evaluated from scratch using Tensorflow. It also supports real-time recognition of sign language gestures using a webcam.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.x)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Getting Started

Follow the steps below to set up and run the project:

1. Clone the repository:
   ```shell
   git clone https://github.com/AbtgCoder/Real-Time-Sign-Language-Detection.git
   ```

2. Navigate to the repository folder:
   ```shell
   cd Real-Time-Sign-Language-Detection
   ```

3. Include the Prebuilt Model:
   -  Place your prebuilt Siamese Network model file (`sign_gesture_model.h5`) in the root folder of the repository.

4. Train the Model:
   - Run the `main.py` script to prepare the datasets, train the model, and evaluate its performance on test data.
   ```shell
   python main.py
   ```

5. Real-Time Sign Language Recognition:
   - Run the `real_time_gesture_recognition.py` script to perform real-time sign language recognition using the trained model.
   ```shell
   python real_time_face_rec.py
   ```

## File Descriptions

- `main.py`: The main script to run the sign language recognition program.
- `mediapipe_utils.py`: Utility functions for using MediaPipe to generate   estimate landmarks for hands, face and pose.
- `data_processing.py`: Script to collect, store training data and prepare datasets for training of the model.
- `model_training.py`: Script to train a deep learning model on the collected data.
- `gesture_recognition.py`: Script to perform real-time gesture recognition using the trained model.
- `sign_gesture_model.h5`: Pre-trained model file for sign language recognition.

## Data Folder
The `data` folder contains the training data collected. 
It is structured as follows:

- `data/[action]/[sequence]/[frame].npy`: Images of various gestures, where:
  - `[action]`: Name of the gesture/action (e.g., hello, thanks, iloveyou).
  - `[sequence]`: Sequence number of the gesture (multiple sequences can be collected for each gesture).
  - `[frame]`: Frame number of the gesture sequence.


## Results and Evaluation

After training and evaluation, the model's performance will be displayed, including the multilabel confusion matrix, and the accuracy score. The results will give insights into the model's ability to detect and recognize different sign language gestures.

## License

[MIT License](LICENSE.txt)

The project is open source and released under the terms of the MIT License. See the [LICENSE](LICENSE.txt) file for more details.

## Contact

For any questions or inquiries, you can reach me at:
- Email:  [abtgofficial@gmail.com](mailto:abtgofficial@gmail.com)

