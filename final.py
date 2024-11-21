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
import utils
import serial

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins setup
motor1_ENA = 17
motor1_IN1 = 23
motor1_IN2 = 24
motor2_ENA = 16
motor2_IN1 = 20
motor2_IN2 = 21

GPIO.setup(motor1_ENA, GPIO.OUT)
GPIO.setup(motor1_IN1, GPIO.OUT)
GPIO.setup(motor1_IN2, GPIO.OUT)
GPIO.setup(motor2_ENA, GPIO.OUT)
GPIO.setup(motor2_IN1, GPIO.OUT)
GPIO.setup(motor2_IN2, GPIO.OUT)

# Enable PWM on the ENA pins
pwm_motor1 = GPIO.PWM(motor1_ENA, 100)
pwm_motor2 = GPIO.PWM(motor2_ENA, 100)
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

# Variables for tracking and SMS
tracking_enabled = False
last_bbox_size = None
last_sms_time = 0  # Timestamp of the last SMS sent
sms_interval = 180  # 180 seconds (3 minutes) interval between SMS notifications

# GSM Module configuration
ser = serial.Serial("/dev/ttyS0", 115200)
ser.flushInput()
phone_number = '+27718948504'
text_message = 'An intruder has been detected!!'
power_key = 6

# Motor control functions
def move_forward(speed):
    GPIO.output(motor1_IN1, GPIO.HIGH)
    GPIO.output(motor1_IN2, GPIO.LOW)
    GPIO.output(motor2_IN1, GPIO.LOW)
    GPIO.output(motor2_IN2, GPIO.HIGH)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Moving forward with speed {speed}")

def move_backward(speed):
    GPIO.output(motor1_IN1, GPIO.LOW)
    GPIO.output(motor1_IN2, GPIO.HIGH)
    GPIO.output(motor2_IN1, GPIO.HIGH)
    GPIO.output(motor2_IN2, GPIO.LOW)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Moving backward with speed {speed}")

def turn_left(speed):
    GPIO.output(motor1_IN1, GPIO.HIGH)
    GPIO.output(motor1_IN2, GPIO.LOW)
    GPIO.output(motor2_IN1, GPIO.HIGH)
    GPIO.output(motor2_IN2, GPIO.LOW)
    pwm_motor1.ChangeDutyCycle(speed)
    pwm_motor2.ChangeDutyCycle(speed)
    print(f"Turning left with speed {speed}")

def turn_right(speed):
    GPIO.output(motor1_IN1, GPIO.LOW)
    GPIO.output(motor1_IN2, GPIO.HIGH)
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

# SMS sending functions with threading
def send_sms_in_thread(phone_number, text_message):
    global last_sms_time
    current_time = time.time()
    if current_time - last_sms_time >= sms_interval:  # Check if 3 minutes have passed
        sms_thread = threading.Thread(target=send_sms_worker, args=(phone_number, text_message))
        sms_thread.start()
    else:
        time_remaining = int(sms_interval - (current_time - last_sms_time))
        print(f"SMS buffer active. Next SMS can be sent in {time_remaining} seconds.")

def send_sms_worker(phone_number, text_message):
    global last_sms_time
    print("Sending SMS alert...")
    try:
        send_at("AT+CMGF=1", "OK", 1)
        ser.write(f'AT+CMGS="{phone_number}"\r\n'.encode())
        time.sleep(2)
        ser.write((text_message + "\x1A").encode())
        send_at('', "OK", 20)
        last_sms_time = time.time()  # Update the last SMS timestamp after successful send
        print("SMS sent successfully.")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def send_at(command, back, timeout):
    ser.write((command + '\r\n').encode())
    time.sleep(timeout)
    if ser.inWaiting():
        rec_buff = ser.read(ser.inWaiting())
        return back in rec_buff.decode()
    return False

def power_on(power_key):
    print("Powering on GSM module...")
    GPIO.setup(power_key, GPIO.OUT)
    GPIO.output(power_key, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(power_key, GPIO.LOW)
    time.sleep(20)
    ser.flushInput()

# Robot control and object detection
def control_robot():
    global tracking_enabled
    while True:
        pygame.event.pump()
        axis_0 = joystick.get_axis(0)
        axis_3 = joystick.get_axis(3)
        if joystick.get_button(0):  # A button
            tracking_enabled = True
            print("Tracking enabled")
        if joystick.get_button(1):  # B button
            tracking_enabled = False
            print("Tracking disabled")
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

def run_object_detection(model, camera_id, width, height, num_threads, enable_edgetpu):
    global tracking_enabled, last_bbox_size
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Configure the object detector
    options = vision.ObjectDetectorOptions(
        base_options=core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads),
        detection_options=processor.DetectionOptions(max_results=1, score_threshold=0.5)
    )
    detector = vision.ObjectDetector.create_from_options(options)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit("ERROR: Unable to read from webcam.")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect(vision.TensorImage.create_from_array(rgb_image))
        
        # Filter results to detect 'person'
        filtered_results = [d for d in detection_result.detections if d.categories[0].category_name == 'person']
        
        if filtered_results and tracking_enabled:
            send_sms_in_thread(phone_number, text_message)  # Send SMS if a person is detected
            
            detection = filtered_results[0]
            bbox = detection.bounding_box
            bbox_size = bbox.width * bbox.height
            
            # Initialize last_bbox_size if it's None
            if last_bbox_size is None:
                last_bbox_size = bbox_size

            # Movement decisions based on bounding box size changes
            if bbox_size > last_bbox_size * 1.1:  # Person is moving closer (increased bounding box size)
                move_backward(50)  # Move backward
                print("Person is approaching; moving backward.")
            elif bbox_size < last_bbox_size * 0.9:  # Person is moving away (decreased bounding box size)
                move_forward(50)  # Move forward
                print("Person is moving away; moving forward.")
            else:
                stop()  # Stop if the person is standing still
                print("Person is stationary; stopping.")

            # Update last bounding box size for the next frame
            last_bbox_size = bbox_size
        else:
            # Stop if no person is detected while tracking is enabled
            if tracking_enabled:
                stop()

        # Display the frame with detections
        image = utils.visualize(image, filtered_results)
        cv2.imshow("object_detector", image)

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientdet_lite0.tflite")
    parser.add_argument("--cameraId", type=int, default=0)
    parser.add_argument("--frameWidth", type=int, default=640)
    parser.add_argument("--frameHeight", type=int, default=480)
    parser.add_argument("--numThreads", type=int, default=4)
    parser.add_argument("--enableEdgeTPU", action="store_true", default=False)
    args = parser.parse_args()
    power_on(power_key)
    t1 = threading.Thread(target=run_object_detection, args=(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU))
    t2 = threading.Thread(target=control_robot)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == "__main__":
    main()

