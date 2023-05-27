from mediapipe_utils import mp_holistic, mediapipe_detection, draw_landmarks, draw_styled_landmarks

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



DATA_PATH = os.path.join("MP_Data")
actions = np.array(["hello", "thanks", "iloveyou"])
no_sequences = 30
sequence_length = 30
threshold = 0.8

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame
def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(132)
    )

    face = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.face_landmarks.landmark
            ]
        ).flatten()
        if results.face_landmarks
        else np.zeros(1872)
    )

    lh = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.left_hand_landmarks.landmark
            ]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(84)
    )

    rh = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.right_hand_landmarks.landmark
            ]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(84)
    )

    return np.concatenate([pose, face, lh, rh])

def create_directories():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def capture_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    cap.set(cv2.CAP_PROP_ZOOM, 1)

    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass


    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    _, frame = cap.read()

                    image, results = mediapipe_detection(frame, holistic)

                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(
                            image,
                            "STARTING COLLECTION",
                            (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            4,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            f"Collecting frames for {action} Video Number {sequence}",
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(
                            image,
                            f"Collecting frames for {action} Video Number {sequence}",
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(
                        DATA_PATH, action, str(sequence), str(frame_num)
                    )
                    np.save(npy_path, keypoints)

                    cv2.imshow("OpenCV Feed", image)

                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break
    cap.release()
    cv2.destroyAllWindows()

def prepare_dataset():
    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(
                    os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                )
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    return X_train, X_test, y_train, y_test
