import pygame
import sqlite3
import json
import numpy as np
from typing import List, Tuple

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

class TrajectoryVisualizer:
    def __init__(self, db_path: str, window_size: Tuple[int, int] = (1200, 600)):
        """Initialize the visualizer with pygame setup"""
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Vehicle Trajectory Visualization")
        
        self.db_path = db_path
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Define coordinate transformation parameters for both views
        view_width = window_size[0] // 2
        view_height = window_size[1]
        
        # Calculate scale for normal view based on the desired area
        x_range = 550 - 80  # = 470
        y_range = 240 - (-20)  # = 260
        x_scale = view_width / x_range
        y_scale = view_height / y_range
        normal_scale = min(x_scale, y_scale)  # Use the smaller scale to fit both dimensions
        
        self.normal_view = {
            'scale': normal_scale,
            'offset_x': -80 * normal_scale,  # Shift to start x from 80
            'offset_y': 420 * normal_scale,  # Shift to handle y range
            'surface': pygame.Surface((view_width, view_height)),
            'fixed_camera': True
        }
        
        self.zoomed_view = {
            'scale': 4.0,
            'offset_x': view_width // 2,
            'offset_y': view_height // 2,
            'surface': pygame.Surface((view_width, view_height)),
            'fixed_camera': False
        }
        
        # Vehicle dimensions (in meters)
        self.vehicle_length = 4.5
        self.vehicle_width = 2.0
        
        # Adjust movement parameters
        self.key_pressed_time = 0
        self.frame_delay = 30  # Reduced from 100 to 30 milliseconds
        self.frame_skip = 3    # Add frame skip to move faster
        self.last_frame_change = 0

    def world_to_screen(self, x: float, y: float, ego_x: float, ego_y: float, view: dict) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        if view['fixed_camera']:
            # Fixed camera view - no ego vehicle offset
            screen_x = int(x * view['scale'] + view['offset_x'])
            screen_y = int(-y * view['scale'] + view['offset_y'])
        else:
            # Ego-centered view
            rel_x = x - ego_x
            rel_y = y - ego_y
            screen_x = int(rel_x * view['scale'] + view['offset_x'])
            screen_y = int(-rel_y * view['scale'] + view['offset_y'])
        return (screen_x, screen_y)

    def draw_vehicle(self, x: float, y: float, yaw: float, ego_x: float, ego_y: float, 
                    view: dict, color: Tuple[int, int, int] = RED, is_ego: bool = False):
        """Draw a rectangle representing the vehicle with orientation"""
        center = self.world_to_screen(x, y, ego_x, ego_y, view)
        
        # Calculate corners of the rectangle
        corners = []
        length = self.vehicle_length * view['scale']
        width = self.vehicle_width * view['scale']
        
        # Calculate rotated corners
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Front right corner
        dx = length/2 * cos_yaw - width/2 * sin_yaw
        dy = length/2 * sin_yaw + width/2 * cos_yaw
        corners.append((center[0] + dx, center[1] - dy))
        
        # Front left corner
        dx = length/2 * cos_yaw + width/2 * sin_yaw
        dy = length/2 * sin_yaw - width/2 * cos_yaw
        corners.append((center[0] + dx, center[1] - dy))
        
        # Rear left corner
        dx = -length/2 * cos_yaw + width/2 * sin_yaw
        dy = -length/2 * sin_yaw - width/2 * cos_yaw
        corners.append((center[0] + dx, center[1] - dy))
        
        # Rear right corner
        dx = -length/2 * cos_yaw - width/2 * sin_yaw
        dy = -length/2 * sin_yaw + width/2 * cos_yaw
        corners.append((center[0] + dx, center[1] - dy))
        
        # Draw the vehicle rectangle
        color = GREEN if is_ego else color
        pygame.draw.polygon(view['surface'], color, corners)
        
        # Draw a line indicating the front of the vehicle
        front_center = (
            center[0] + (length/2) * cos_yaw,
            center[1] - (length/2) * sin_yaw
        )
        pygame.draw.line(view['surface'], BLACK, center, front_center, 2)

    def draw_trajectory(self, trajectory: List[Tuple[float, float]], ego_x: float, ego_y: float,
                       view: dict, color: Tuple[int, int, int] = BLUE):
        """Draw the predicted trajectory"""
        if len(trajectory) < 2:
            return
        screen_points = [self.world_to_screen(x, y, ego_x, ego_y, view) for x, y in trajectory]
        pygame.draw.lines(view['surface'], color, False, screen_points, 2)

    def handle_continuous_movement(self, current_frame: int, timestamps: List[int]) -> int:
        """Handle continuous movement when keys are held down"""
        keys = pygame.key.get_pressed()
        current_time = pygame.time.get_ticks()
        
        if current_time - self.last_frame_change > self.frame_delay:
            if keys[pygame.K_RIGHT]:
                current_frame = (current_frame + self.frame_skip) % len(timestamps)
                self.last_frame_change = current_time
            elif keys[pygame.K_LEFT]:
                current_frame = (current_frame - self.frame_skip) % len(timestamps)
                self.last_frame_change = current_time
            
        return current_frame

    def run(self):
        """Main visualization loop"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Get all unique timestamps
        cur.execute("SELECT DISTINCT frame FROM predict_traj ORDER BY frame")
        timestamps = [row[0] for row in cur.fetchall()]
        
        current_frame = 0
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # Handle continuous movement
            current_frame = self.handle_continuous_movement(current_frame, timestamps)

            # Clear both surfaces
            self.normal_view['surface'].fill(WHITE)
            self.zoomed_view['surface'].fill(WHITE)
            
            # Get data for current timestamp
            cur.execute("""
                SELECT vehicle_id, x, y, p_traj, vel 
                FROM predict_traj 
                WHERE frame=?
            """, (timestamps[current_frame],))
            
            frame_data = cur.fetchall()
            
            if frame_data:
                ego_x, ego_y = frame_data[0][1], frame_data[0][2]
                
                # Draw in both views
                for view in [self.normal_view, self.zoomed_view]:
                    for i, (vehicle_id, x, y, p_traj, vel) in enumerate(frame_data):
                        trajectory = json.loads(p_traj)
                        
                        # Calculate yaw from trajectory
                        if len(trajectory) > 1:
                            dx = trajectory[1][0] - x
                            dy = trajectory[1][1] - y
                            yaw = np.arctan2(dy, dx)
                        else:
                            yaw = 0
                        
                        is_ego = (i == 0)
                        self.draw_trajectory(trajectory, ego_x, ego_y, view)
                        self.draw_vehicle(x, y, yaw, ego_x, ego_y, view, is_ego=is_ego)
                
                # Blit both surfaces to the screen
                self.screen.blit(self.normal_view['surface'], (0, 0))
                self.screen.blit(self.zoomed_view['surface'], (self.window_size[0]//2, 0))
                
                # Draw dividing line
                pygame.draw.line(self.screen, BLACK, 
                               (self.window_size[0]//2, 0),
                               (self.window_size[0]//2, self.window_size[1]), 2)
            
            # Draw frame number
            font = pygame.font.Font(None, 36)
            text = font.render(f"Frame: {timestamps[current_frame]}", True, BLACK)
            self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        conn.close()

if __name__ == "__main__":
    DB_PATH = "/Users/miakho/Code/LimSim/detector.db"
    visualizer = TrajectoryVisualizer(DB_PATH, (1200, 600))
    visualizer.run() 