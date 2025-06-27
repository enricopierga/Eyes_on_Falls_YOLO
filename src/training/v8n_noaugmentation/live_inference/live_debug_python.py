import cv2
import numpy as np
import tensorflow as tf
import time

# Configurazione
MODEL_PATH = "best_fall_detection_yolo11_float32.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.3
CLASS_NAMES = ["fallen", "not_fallen"]

# Carica TFLite (su PC con TensorFlow full)
tflite = tf.lite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inizializza webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()
    interpreter.invoke()
    end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output = np.squeeze(output_data)  # (N, 6)
    if output.ndim == 1:
        output = np.expand_dims(output, axis=0)  # Caso di singola detection

    boxes = output[:, 0:4]
    scores = output[:, 4]
    class_ids = output[:, 5].astype(int)

    # Filtra confidenza
    mask = scores > CONF_THRESHOLD
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    # Adatta box a dimensione originale
    scale_x = frame.shape[1] / INPUT_SIZE
    scale_y = frame.shape[0] / INPUT_SIZE

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        # Protezione robusta su class index
        if class_ids[i] >= len(CLASS_NAMES):
            label = f"Unknown: {scores[i]:.2f}"
        else:
            label = f"{CLASS_NAMES[class_ids[i]]}: {scores[i]:.2f}"

        color = (0, 255, 0) if class_ids[i] == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('YOLOv11 TFLite NMS Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
