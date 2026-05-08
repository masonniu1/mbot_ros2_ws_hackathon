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
        
        # 1. Have we seen a person recently? 
        # Increased to 1.5s to account for the 2Hz camera update rate + processing lag
        time_since_last_seen = time.time() - self.last_seen_time
        if self.person_x is None or time_since_last_seen > 1.5:
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            return

        # 2. Steer towards the person
        error = 320.0 - float(self.person_x)
        
        # --- FIX 1: THE GAIN REDUCTION ---
        # Because we only get 2 updates per second, we must turn SLOWER
        # so we don't sweep past the center between photos.
        # Reduced Kp gain from 0.003 to 0.0012
        turn_speed = error * 0.0012  
        
        # Apply your steering inverter here (1.0 or -1.0 depending on what worked)
        INVERT_STEERING = -1.0  
        
        # Clamp maximum turn speed to a lower value so it doesn't aggressively whip
        cmd.angular.z = max(-0.4, min(0.4, turn_speed)) * INVERT_STEERING

        # --- FIX 2: THE SMOOTH ARC APPROACH ---
        #/home/mbot/mbot_ws/src/companion_behavior/companion_behavior/brain_node.py
        # 3. Move forward AND steer at the same time
        if self.front_distance <= 0.75: 
            # We have arrived! 1 meter away.
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0 # Force a stop to keep the camera steady
            # TODO: Trigger Gemini API Voice here!
        else:
            # Base approach speed
            approach_speed = 0.15 
            
            # The "Speed Penalty" Hack:
            # If the person is at the extreme edge of the screen (error is large),
            # this calculation naturally slows the forward speed down so it has 
            # time to carve a sharp turn. If they are perfectly centered (error = 0),
            # it drives at full approach speed!
            speed_penalty = abs(error) / 320.0 
            cmd.linear.x = approach_speed * (1.0 - speed_penalty)
            
            # Safety clamp so it never tries to drive backwards
            cmd.linear.x = max(0.0, cmd.linear.x)

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = CompanionBrain()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()