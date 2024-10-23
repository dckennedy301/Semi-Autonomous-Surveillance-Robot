import argparse
import sys
import time
import cv2
import threading
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import RPi.GPIO as GPIO
import pygame
import utils1

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor 1 pins
motor1_ENA = 17  # Pin 11
motor1_IN1 = 23  # Pin 16
motor1_IN2 = 24  # Pin 18

# Motor 2 pins
motor2_ENA = 16  # Pin 36
motor2_IN1 = 20  # Pin 38
motor2_IN2 = 21  # Pin 40

# Setup GPIO pins as output
GPIO.setup(motor1_ENA, GPIO.OUT)
GPIO.setup(motor1_IN1, GPIO.OUT)
GPIO.setup(motor1_IN2, GPIO.OUT)

GPIO.setup(motor2_ENA, GPIO.OUT)
GPIO.setup(motor2_IN1, GPIO.OUT)
GPIO.setup(motor2_IN2, GPIO.OUT)

# Enable PWM on the ENA pins
pwm_motor1 = GPIO.PWM(motor1_ENA, 100)  # 100 Hz frequency
pwm_motor2 = GPIO.PWM(motor2_ENA, 100)  # 100 Hz frequency

pwm_motor1.start(0)
pwm_motor2.start(0)

# Initialize Pygame and Xbox controller
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

# Variable to store tracking state
tracking_enabled = False
last_bbox_size = None  # Store the last bounding box size to detect movement


def move_forward(speed):
    GPIO.output(motor1_IN1, GPIO.HIGH)
    GPIO.output(motor1_IN2, GPIO.LOW)
    GPIO.output(motor2_IN1, GPIO.HIGH)
    GPIO.output(motor2_IN2, GPIO.LOW)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Moving forward with speed {speed}")


def move_backward(speed):
    GPIO.output(motor1_IN1, GPIO.LOW)
    GPIO.output(motor1_IN2, GPIO.HIGH)
    GPIO.output(motor2_IN1, GPIO.LOW)
    GPIO.output(motor2_IN2, GPIO.HIGH)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Moving backward with speed {speed}")


def turn_left(speed):
    GPIO.output(motor1_IN1, GPIO.LOW)
    GPIO.output(motor1_IN2, GPIO.HIGH)
    GPIO.output(motor2_IN1, GPIO.HIGH)
    GPIO.output(motor2_IN2, GPIO.LOW)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Turning left with speed {speed}")


def turn_right(speed):
    GPIO.output(motor1_IN1, GPIO.HIGH)
    GPIO.output(motor1_IN2, GPIO.LOW)
    GPIO.output(motor2_IN1, GPIO.LOW)
    GPIO.output(motor2_IN2, GPIO.HIGH)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Turning right with speed {speed}")


def stop():
    GPIO.output(motor1_IN1, GPIO.LOW)
    GPIO.output(motor1_IN2, GPIO.LOW)
    GPIO.output(motor2_IN1, GPIO.LOW)
    GPIO.output(motor2_IN2, GPIO.LOW)
    pwm_motor1.ChangeDutyCycle(0)
    pwm_motor2.ChangeDutyCycle(0)
    print("Stopping")


def control_robot():
    global tracking_enabled
    try:
        while True:
            pygame.event.pump()
            axis_0 = joystick.get_axis(0)
            axis_3 = joystick.get_axis(3)

            # A button for tracking, B button for manual control
            if joystick.get_button(0):  # A button (start tracking)
                tracking_enabled = True
                print("Tracking enabled")
            if joystick.get_button(1):  # B button (stop tracking)
                tracking_enabled = False
                print("Tracking disabled")

            # Manual control if tracking is disabled
            if not tracking_enabled:
                if axis_3 < -0.1:
                    move_forward(int(-axis_3 * 100))
                elif axis_3 > 0.1:
                    move_backward(int(axis_3 * 100))
                else:
                    stop()

                if axis_0 < -0.1:
                    turn_left(int(-axis_0 * 100))
                elif axis_0 > 0.1:
                    turn_right(int(axis_0 * 100))

            time.sleep(0.1)
    finally:
        stop()
        pygame.quit()
        GPIO.cleanup()


def run_object_detection(model, camera_id, width, height, num_threads, enable_edgetpu):
    global tracking_enabled, last_bbox_size
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=1, score_threshold=0.5)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(input_tensor)

        filtered_results = [d for d in detection_result.detections if d.categories[0].category_name == 'person']

        if filtered_results and tracking_enabled:
            detection = filtered_results[0]
            bbox = detection.bounding_box
            x_center = (bbox.origin_x + bbox.width // 2)
            bbox_size = bbox.width * bbox.height  # Calculate bounding box area to track size changes

            if last_bbox_size is None:
                last_bbox_size = bbox_size

            # Movement decision based on size change
            if bbox_size < last_bbox_size * 0.9:  # Person is moving closer
                move_backward(50)  # Move backward when the person comes closer
            elif bbox_size > last_bbox_size * 1.1:  # Person is moving away
                move_forward(50)  # Move forward when the person moves away
            else:
                stop()  # Stop if the person is standing still

            # Adjust turning based on person's position in the frame
            if x_center < width // 3:
                turn_left(50)
            elif x_center > 2 * width // 3:
                turn_right(50)

            # Update last bounding box size for the next frame
            last_bbox_size = bbox_size

        else:
            if tracking_enabled:
                stop()

        image = utils1.visualize(image, filtered_results)
        cv2.imshow('object_detector', image)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', type=int, default=0)
    parser.add_argument('--frameWidth', type=int, default=640)
    parser.add_argument('--frameHeight', type=int, default=480)
    parser.add_argument('--numThreads', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', action='store_true', default=False)
    args = parser.parse_args()

    t1 = threading.Thread(target=run_object_detection, args=(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU))
    t2 = threading.Thread(target=control_robot)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == '__main__':
    main()
