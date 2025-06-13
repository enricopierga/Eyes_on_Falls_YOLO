import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Configurazione
MODEL_PATH = "v8n_dataset_pierga_float16.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
CLASS_NAMES = ["fallen", "not_fallen"]

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def process_output(output):
    output = np.squeeze(output).transpose()  # (8400, 6)
    boxes = sigmoid(output[:, 0:4])
    scores = sigmoid(output[:, 4])
    class_scores = sigmoid(output[:, 5:])
    class_ids = np.argmax(class_scores, axis=-1)
    class_conf = np.max(class_scores, axis=-1)
    final_scores = scores * class_conf
    mask = final_scores > CONF_THRESHOLD
    boxes, final_scores, class_ids = boxes[mask], final_scores[mask], class_ids[mask]
    boxes = xywh2xyxy(boxes) * INPUT_SIZE
    return boxes, final_scores, class_ids

# Video capture
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
    boxes, scores, class_ids = process_output(output_data)

    scale_x = frame.shape[1] / INPUT_SIZE
    scale_y = frame.shape[0] / INPUT_SIZE

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
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('YOLOv8 TFLite Live', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
