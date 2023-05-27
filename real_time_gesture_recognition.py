from mediapipe_utils import mp_holistic, mediapipe_detection, draw_landmarks, draw_styled_landmarks
from data_processing import actions, extract_keypoints


import cv2
import numpy as np

def run_gesture_recognition(cap, model):
    
    sequence = []
    sentence = []
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
    
    detection_confidence = 0.5
    tracking_confidence = 0.5
    
    with mp_holistic.Holistic(
        min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence
    ) as holistic:
        while cap.isOpened():
            _, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                image = prob_viz(res, actions, image, colors)

            if len(sentence) > 5:
                sentence = sentence[-5:]

            cv2.imshow("Live Feed", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
