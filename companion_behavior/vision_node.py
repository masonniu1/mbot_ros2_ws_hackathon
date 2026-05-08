import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
import os
from ultralytics import YOLO

class CompanionVision(Node):
    def __init__(self):
        super().__init__('companion_vision')
        
        # Publisher to send the person's location to the brain
        self.target_pub = self.create_publisher(Point, '/person_target', 10)
        
        # Load the lightweight model
        self.model = YOLO("yolo11n.pt") 
        self.image_path = "/dev/shm/companion_eye.jpg"
        
        # Run the detection loop at 5 Hz (every 0.2 seconds)
        self.timer = self.create_timer(0.2, self.detect_person)
        self.get_logger().info("Vision Node Started. Looking for elderly companion...")

    def detect_person(self):
        if not os.path.exists(self.image_path):
            return

        frame = cv2.imread(self.image_path)
        if frame is None:
            return

        results = self.model(frame, imgsz=320, conf=0.5, verbose=False)

        for box in results[0].boxes:
            if results[0].names[int(box.cls[0])] == "person":
                # Calculate the center of the person
                x_min, _, x_max, _ = box.xyxy[0].tolist()
                center_x = (x_min + x_max) / 2
                
                # Publish the location!
                msg = Point()
                msg.x = float(center_x)     # Pixel location (0 to 640)
                msg.z = float(box.conf[0])  # Confidence score
                self.target_pub.publish(msg)
                
                self.get_logger().info(f"Person spotted at X: {center_x:.0f}")
                return # Stop after finding the first person

def main(args=None):
    rclpy.init(args=args)
    node = CompanionVision()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()