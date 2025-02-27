# idea was done by Mathieu Dijkstra under open source license!
# i emagen that there is a sphere the core emotions at the outher edg and in the center neutral. radius is 100 of that spere. 
# so like 
#  "joy": "sadness" outher right and outher left
#  "anger": "trust" outher front and outher back
#  "fear": "anticipation" outher up and outher down
#  "disgust": "surprise", also some where in that spere outher .
# so for every emotion there is a position (perfect at same distances from eachother.
# also like Neuro Languistic Programing there is a modalitie for what must ly close to eachoter
# this is just the chat with a ai not realey yet worked out  but its just a core idea!!!!
#
# using this sphere is by positioning a point in that spere where the human is most likley at
# 
# for audio(hearing), video(seeing), typing(typing speed changes) & mouse, feeling in the future. every modal has a own spehere. by add the speres together they get a visual (location in the spere) what emotion the human is for better understanding hem/her!!! 
#
# for making future robots AI that understand hamans more!
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class EmotionalSphere:
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0])  # Neutral center
        self.emotions = {
            "joy": self._spherical_to_cartesian(0, 90, 100),
            "sadness": self._spherical_to_cartesian(180, 90, 100),
            "anger": self._spherical_to_cartesian(90, 90, 100),
            "fear": self._spherical_to_cartesian(270, 90, 100),
            "trust": self._spherical_to_cartesian(0, 0, 100),
            "disgust": self._spherical_to_cartesian(180, 180, 100),
            "surprise": self._spherical_to_cartesian(45, 90, 100),
            "anticipation": self._spherical_to_cartesian(315, 90, 100),
        }

    @staticmethod
    def _spherical_to_cartesian(theta, phi, radius):
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        x = radius * np.sin(phi_rad) * np.cos(theta_rad)
        y = radius * np.sin(phi_rad) * np.sin(theta_rad)
        z = radius * np.cos(phi_rad)
        return np.array([x, y, z])

    def get_opposite_emotion(self, emotion):
        opposites = {
            "joy": "sadness",
            "sadness": "joy",
            "anger": "trust",
            "trust": "anger",
            "fear": "anticipation",
            "anticipation": "fear",
            "disgust": "surprise",
            "surprise": "disgust"
        }
        return opposites.get(emotion)

    def update_state(self, target_emotion, learning_rate=0.1):
        target = self.emotions[target_emotion]
        current = self.state
        current_radius = np.linalg.norm(current)

        if current_radius == 0:
            self.state = target * learning_rate
            return

        current_dir = current / current_radius
        target_dir = target / 100.0
        angle = np.arccos(np.clip(np.dot(current_dir, target_dir), -1, 1))

        # Speed and intensity control
        speed_factor = 1 - np.exp(-5 * current_radius / 100)
        speed = learning_rate * angle * speed_factor

        # Adjust intensity based on the angle
        intensity_increase = learning_rate * 10 * angle / np.pi
        new_radius = min(current_radius + intensity_increase, 100)

        # Boost speed for opposite emotions
        opposite_emotion = self.get_opposite_emotion(target_emotion)
        if opposite_emotion:
            opposite_target = self.emotions[opposite_emotion]
            opposite_angle = np.arccos(np.clip(np.dot(current_dir, opposite_target / 100.0), -1, 1))
            speed *= 1 + (opposite_angle * 0.5)

        # Spherical interpolation
        t = min(speed, 1.0)
        new_dir = current_dir * (1 - t) + target_dir * t
        new_dir /= np.linalg.norm(new_dir)

        self.state = new_dir * new_radius

    def display_state(self):
        radius = np.linalg.norm(self.state)
        if radius == 0:
            print("Neutral (0, 0, 0)")
            return

        closest_emotion = min(
            self.emotions.items(),
            key=lambda x: np.linalg.norm(self.state - x[1])
        )
        print(f"State: {np.round(self.state, 2)}")
        print(f"Intensity: {radius:.2f}/100 | Closest Emotion: {closest_emotion[0]}")

    def visualize(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for emotion, position in self.emotions.items():
            color, alpha = self._get_color_and_alpha(emotion)
            ax.scatter(*position, color=color, alpha=alpha, s=100)
            ax.text(*position, emotion, color=color)

        ax.scatter(*self.state, color='purple', s=200, label='Current State')
        ax.legend()

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

    def _get_color_and_alpha(self, emotion):
        color_map = {
            "joy": (0, 1, 0, 0.8),         # Green for positive
            "sadness": (1, 0, 0, 0.8),     # Red for negative
            "anger": (1, 0.5, 0, 0.8),     # Orange for high energy
            "fear": (0.5, 0, 1, 0.8),      # Purple for fear
            "trust": (0, 0.5, 1, 0.8),     # Blue for trust
            "disgust": (0.6, 0.2, 0, 0.8), # Brown for disgust
            "surprise": (1, 1, 0, 0.8),    # Yellow for surprise
            "anticipation": (1, 0, 1, 0.8) # Magenta for anticipation
        }
        return color_map.get(emotion, (0.5, 0.5, 0.5, 0.5))


# Example Usage
sphere = EmotionalSphere()

sphere.update_state("joy")
sphere.display_state()
sphere.visualize()

sphere.update_state("sadness")
sphere.display_state()
sphere.visualize()

for _ in range(5):
    sphere.state *= 0.8
sphere.display_state()
sphere.visualize()
