import cv2

def detect_license_plates():
    # Load the pre-trained cascade classifier
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # Initialize the webcam (camera index 0)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    min_area = 500
    count = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to capture video.")
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect license plates
        plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                img_roi = img[y:y + h, x:x + w]
                cv2.imshow("ROI", img_roi)

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f"Plates/scaned_img_{count}.jpg", img_roi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results", img)
            cv2.waitKey(500)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_license_plates()
