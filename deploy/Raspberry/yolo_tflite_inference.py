import cv2
import numpy as np
import tensorflow as tf
import time

# Configurazione
MODEL_PATH = "best_fall_detection_yolo11_float16.tflite"
INPUT_IMAGE_SIZE = 640
CONF_THRESHOLD = 0.3
CLASS_NAMES = ["fallen", "not_fallen"]

# Usa TensorFlow completo (PC)
tflite = tf.lite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Normalizzazione output helper
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Parser semplificato per export Ultralytics TFLite (con decode già incluso nel file)
def process_output(output):
    output = np.squeeze(output).transpose()  # (8400, 6)
    boxes = output[:, 0:4]
    scores = sigmoid(output[:, 4])
    class_scores = sigmoid(output[:, 5:])
    class_ids = np.argmax(class_scores, axis=-1)
    class_conf = np.max(class_scores, axis=-1)
    final_scores = scores * class_conf

    mask = final_scores > CONF_THRESHOLD
    boxes, final_scores, class_ids = boxes[mask], final_scores[mask], class_ids[mask]

    # Se necessario riscalare le box a INPUT_SIZE (ma spesso già lo sono)
    return boxes, final_scores, class_ids

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_IMAGE_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_IMAGE_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_resized = cv2.resize(frame, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()
    interpreter.invoke()
    end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    boxes, scores, class_ids = process_output(output_data)

    # Ridimensionamento box su immagine originale
    scale_x = frame.shape[1] / INPUT_IMAGE_SIZE
    scale_y = frame.shape[0] / INPUT_IMAGE_SIZE

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f"{CLASS_NAMES[class_ids[i]]}: {scores[i]:.2f}"
        color = (0, 255, 0) if class_ids[i] == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('YOLOv11 TFLite Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
