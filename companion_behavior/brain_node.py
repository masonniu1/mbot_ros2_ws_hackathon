import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
import time

class CompanionBrain(Node):
    def __init__(self):
        super().__init__('companion_brain')
        
        self.vision_sub = self.create_subscription(Point, '/person_target', self.vision_cb, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.person_x = None
        self.last_seen_time = 0.0
        self.front_distance = 999.0  
        
        # Control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

    def vision_cb(self, msg):
        self.person_x = msg.x
        self.last_seen_time = time.time()

    def lidar_cb(self, msg):
        # Look at the lasers directly in front (-15 to +15 degrees)
        front_ranges = msg.ranges[:15] + msg.ranges[-15:]
        valid_ranges = [r for r in front_ranges if 0.1 < r < 10.0]
        
        if valid_ranges:
            self.front_distance = min(valid_ranges)
        else:
            self.front_distance = 999.0

    def control_loop(self):
        cmd = Twist()
        
        # 1. Have we seen a person recently? (Within the last 1 second)
        if self.person_x is None or (time.time() - self.last_seen_time > 1.0):
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            return

        # 2. Steer towards the person (Center of a 640px image is 320)
        error = 320 - self.person_x
        turn_speed = error * 0.003  # Kp Gain: Adjust this to turn faster/slower
        
        # Clamp maximum turn speed
        cmd.angular.z = max(-0.8, min(0.8, turn_speed))

        # 3. Move forward if they are far, stop if they are close
        if abs(error) < 150: # Only drive forward if they are mostly centered
            if self.front_distance > 1.0: # 1 meter away
                cmd.linear.x = 0.15 # Drive forward
            else:
                cmd.linear.x = 0.0  # Stop to talk!
                # TODO: Trigger Gemini API Voice here!
        else:
            cmd.linear.x = 0.0 # Pivot in place to center them first

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = CompanionBrain()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()