import cv2
import mediapipe as mp

# Initialize Mediapipe Face Detection and Drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize the Mediapipe Face Detection solution
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and find faces
    results = face_detection.process(image)

    # Convert the image color back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    person_count = 0  # Counter for persons detected

    # Draw face landmarks and count detected faces
    if results.detections:
        person_count = len(results.detections)  # Count the number of faces detected
        for detection in results.detections:
            # Draw bounding boxes around detected faces
            mp_drawing.draw_detection(image, detection)

    # Display the person count on the screen
    cv2.putText(image, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('Head Tracker', image)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
