import json
import os
import queue
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan

import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from vosk import Model, KaldiRecognizer


MODEL_PATH = "mbot_ws/src/companion_behavior/companion_behavior/models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000

# If NoMachine chooses the wrong input, set this to an integer index.
# Run: python -m sounddevice
# Choose a real mic input, NOT anything ending in ".monitor".
INPUT_DEVICE = "default"

# Use a wake word so the robot does not respond to every random sound.
USE_WAKE_WORD = True
WAKE_WORDS = ["robot", "hey robot", "companion", "mbot", "hi robot"]


class CompanionBrain(Node):
    def __init__(self):
        super().__init__("companion_brain")

        self.vision_sub = self.create_subscription(
            Point,
            "/person_target",
            self.vision_cb,
            10,
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_cb,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Following state
        self.person_x = None
        self.last_seen_time = 0.0
        self.front_distance = 999.0

        # Conversation state
        self.conversation_history = []
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.voice_thread_should_run = True
        self.is_generating_response = False

        # Movement tuning from your current follow logic
        self.person_timeout = 1.5
        self.image_center_x = 320.0
        self.turn_gain = 0.0015
        self.invert_steering = -1.0
        self.max_turn_speed = 0.6
        self.arrival_distance = 0.75
        self.approach_speed = 0.3

        # Gemini setup
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
            self.get_logger().info("Gemini client loaded.")
        else:
            self.gemini_client = None
            self.get_logger().warn("Missing GEMINI_API_KEY. Conversation disabled.")

        # Movement loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        # Voice conversation loop in background
        self.voice_thread = threading.Thread(target=self.voice_loop, daemon=True)
        self.voice_thread.start()

        self.get_logger().info("Companion brain started.")
        self.get_logger().info("Following uses vision/LiDAR only.")
        self.get_logger().info("Voice is conversation only, not movement control.")
        self.get_logger().info("Speech output uses espeak-ng/espeak through NoMachine Ubuntu audio.")

    # -------------------------
    # ROS callbacks
    # -------------------------

    def vision_cb(self, msg):
        self.person_x = msg.x
        self.last_seen_time = time.time()

    def lidar_cb(self, msg):
        front_ranges = list(msg.ranges[:15]) + list(msg.ranges[-15:])
        valid_ranges = [r for r in front_ranges if 0.1 < r < 10.0]

        if valid_ranges:
            self.front_distance = min(valid_ranges)
        else:
            self.front_distance = 999.0

    # -------------------------
    # Audio helpers
    # -------------------------

    def audio_callback(self, indata, frames, callback_time, status):
        if status:
            self.get_logger().warn(f"Audio status: {status}")
        self.audio_queue.put(bytes(indata))

    def clear_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def speak(self, text):
        """
        Speak response through the NoMachine-forwarded Ubuntu audio output.

        This script is intended to run on NoMachine Ubuntu / ROS2,
        so we use espeak-ng or espeak instead of macOS 'say'.

        While speaking, microphone processing is ignored so the robot does not
        transcribe its own voice.
        """
        if not text:
            return

        self.is_speaking = True
        self.clear_audio_queue()

        cmd = shutil.which("espeak-ng") or shutil.which("espeak")

        if cmd is None:
            self.is_speaking = False
            self.get_logger().warn("espeak-ng not found. Printing response only.")
            print(f"Robot says: {text}")
            return

        process = subprocess.Popen([cmd, text])
        process.wait()

        # Prevent the mic from hearing the tail end of the robot's own voice.
        time.sleep(0.5)
        self.clear_audio_queue()
        self.is_speaking = False

    # -------------------------
    # Conversation helpers
    # -------------------------

    def has_wake_word(self, text):
        lowered = text.lower().strip()
        return any(wake_word in lowered for wake_word in WAKE_WORDS)

    def remove_wake_word(self, text):
        cleaned = text.lower().strip()

        for wake_word in WAKE_WORDS:
            if cleaned.startswith(wake_word):
                cleaned = cleaned[len(wake_word):].strip()
                break

        return cleaned if cleaned else text

    def ask_gemini_conversation(self, user_text):
        """
        Conversation only. This does NOT return movement commands.
        """
        if self.gemini_client is None:
            return "I can hear you, but Gemini is not configured yet."

        robot_state = {
            "person_x": self.person_x,
            "front_distance_meters": self.front_distance,
            "seconds_since_person_seen": time.time() - self.last_seen_time,
            "behavior": "The robot is autonomously following a person using vision and LiDAR.",
        }

        prompt = f"""
You are the conversation voice for a small friendly companion robot.

The robot is already using its own local vision and LiDAR logic to follow a person.
You do NOT control robot movement.
Do NOT give movement commands.
Do NOT output JSON.

Your job is only to have a short, friendly conversation.

Current robot state:
{json.dumps(robot_state, indent=2)}

Recent conversation:
{json.dumps(self.conversation_history[-8:], indent=2)}

The person said:
"{user_text}"

Reply as the robot in 1 short sentence.
Keep it warm, natural, and under 20 words.
Do not mention Gemini, APIs, implementation details, or that you are an AI model.
"""

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text.strip()

    def handle_user_conversation(self, user_text):
        """
        Run Gemini in a background thread so movement control does not pause.
        """
        if self.is_generating_response:
            self.get_logger().info("Already generating a response; ignoring overlapping speech.")
            return

        self.is_generating_response = True

        def worker():
            try:
                self.get_logger().info(f"Person said: {user_text}")
                print(f"\nPerson said: {user_text}")

                reply = self.ask_gemini_conversation(user_text)

                self.conversation_history.append({
                    "person": user_text,
                    "robot": reply,
                    "time": time.time(),
                })

                self.get_logger().info(f"Robot says: {reply}")
                print(f"Robot says: {reply}\n")

                self.speak(reply)

            except Exception as e:
                self.get_logger().error(f"Conversation failed: {e}")

            finally:
                self.is_generating_response = False

        threading.Thread(target=worker, daemon=True).start()

    def voice_loop(self):
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(
                f"Missing Vosk model at {MODEL_PATH}. Voice conversation disabled."
            )
            return

        try:
            model = Model(MODEL_PATH)
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        except Exception as e:
            self.get_logger().error(f"Failed to load Vosk model: {e}")
            return

        self.get_logger().info("Vosk loaded. Listening for conversation through NoMachine mic.")

        try:
            with sd.RawInputStream(
                device=INPUT_DEVICE,
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback,
            ):
                self.speak("Conversation mode is ready.")

                while self.voice_thread_should_run:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Ignore microphone while robot is speaking.
                    if self.is_speaking:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        user_text = result.get("text", "").strip()

                        if not user_text:
                            continue

                        if USE_WAKE_WORD and not self.has_wake_word(user_text):
                            self.get_logger().info(f"Ignored without wake word: {user_text}")
                            continue

                        if USE_WAKE_WORD:
                            user_text = self.remove_wake_word(user_text)

                        # IMPORTANT:
                        # This is conversation only.
                        # No matter what the person says, we do not change movement logic here.
                        self.handle_user_conversation(user_text)

        except Exception as e:
            self.get_logger().error(f"Voice loop failed: {e}")

    # -------------------------
    # Movement logic: unchanged follow behavior
    # -------------------------

    def control_loop(self):
        cmd = Twist()

        # 1. Have we seen a person recently?
        time_since_last_seen = time.time() - self.last_seen_time

        if self.person_x is None or time_since_last_seen > self.person_timeout:
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            return

        # 2. Steer towards the person
        error = self.image_center_x - float(self.person_x)

        # Turn slowly because camera updates are slow
        turn_speed = error * self.turn_gain

        cmd.angular.z = max(-self.max_turn_speed, min(self.max_turn_speed, turn_speed))
        cmd.angular.z *= self.invert_steering

        # 3. Move forward and steer at the same time
        if self.front_distance <= self.arrival_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            speed_penalty = abs(error) / self.image_center_x
            cmd.linear.x = self.approach_speed * (1.0 - speed_penalty)
            cmd.linear.x = max(0.0, cmd.linear.x)

        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        cmd = Twist()
        cmd.angular.z = 0.0
        cmd.linear.x = 0.0
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CompanionBrain()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.voice_thread_should_run = False
    node.publish_stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()