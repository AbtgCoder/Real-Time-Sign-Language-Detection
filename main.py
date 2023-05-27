from real_time_gesture_recognition import run_gesture_recognition
from model_training import train_model, load_model
from data_processing import prepare_dataset
from model_evaluation import evaluate_model

import cv2

EPOCHS = 1000

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    cap.set(cv2.CAP_PROP_ZOOM, 1)
    
    X_train, X_test, y_train, y_test = prepare_dataset()
    train_model(X_train, y_train, EPOCHS)
    sign_gesture_model = load_model()
    evaluate_model(X_test, y_test, sign_gesture_model)

    run_gesture_recognition(cap, sign_gesture_model)

    cap.release()
    cv2.destroyAllWindows()
