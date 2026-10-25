"""
OptiScan-Agent: Synthetic Industrial PCB & Silicon Inspection Data Generator
Generates realistic golden references and defective boards with optical noise,
glare, micro-cracks, solder bridges, and component misalignments.
"""

import cv2
import numpy as np
import random
from typing import Tuple, Dict, Any, List

class PCBDatasetGenerator:
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height

    def generate_golden_board(self) -> np.ndarray:
        """Generates a perfect, defect-free golden reference PCB."""
        # Dark green substrate
        board = np.full((self.height, self.width, 3), (25, 80, 20), dtype=np.uint8)

        # Ground plane grid pattern
        for y in range(0, self.height, 40):
            cv2.line(board, (0, y), (self.width, y), (28, 92, 23), 1)
        for x in range(0, self.width, 40):
            cv2.line(board, (x, 0), (x, self.height), (28, 92, 23), 1)

        # Copper traces (gold/copper color)
        copper_color = (40, 165, 218)
        trace_paths = [
            [(150, 200), (350, 200), (450, 300), (750, 300), (900, 150)],
            [(150, 240), (330, 240), (430, 340), (730, 340), (880, 190)],
            [(150, 280), (310, 280), (410, 380), (710, 380), (860, 230)],
            [(150, 500), (400, 500), (500, 600), (850, 600), (1050, 450)],
            [(150, 540), (380, 540), (480, 640), (830, 640), (1030, 490)],
            [(200, 700), (600, 700), (700, 600), (1050, 600)]
        ]
        for path in trace_paths:
            for i in range(len(path) - 1):
                cv2.line(board, path[i], path[i+1], copper_color, 4, cv2.LINE_AA)

        # Main Processor IC Chip (center-left)
        ic_center = (450, 450)
        ic_w, ic_h = 160, 160
        ic_top_left = (ic_center[0] - ic_w//2, ic_center[1] - ic_h//2)
        ic_bottom_right = (ic_center[0] + ic_w//2, ic_center[1] + ic_h//2)
        # Chip package body (matte black)
        cv2.rectangle(board, ic_top_left, ic_bottom_right, (35, 35, 35), -1)
        cv2.rectangle(board, ic_top_left, ic_bottom_right, (60, 60, 60), 2)
        # IC silkscreen label
        cv2.putText(board, "ARM-NEON v5", (ic_top_left[0] + 15, ic_top_left[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(board, "AWS-GRAVITON-IO", (ic_top_left[0] + 15, ic_top_left[1] + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.circle(board, (ic_top_left[0] + 15, ic_top_left[1] + 15), 5, (120, 120, 120), -1)

        # IC Pins on all 4 sides (silver/tin solder)
        pin_color = (200, 205, 210)
        num_pins = 12
        spacing = ic_w // (num_pins + 1)
        # Top & Bottom pins
        for i in range(1, num_pins + 1):
            px = ic_top_left[0] + i * spacing
            cv2.rectangle(board, (px - 3, ic_top_left[1] - 18), (px + 3, ic_top_left[1]), pin_color, -1)
            cv2.rectangle(board, (px - 3, ic_bottom_right[1]), (px + 3, ic_bottom_right[1] + 18), pin_color, -1)
        # Left & Right pins
        for i in range(1, num_pins + 1):
            py = ic_top_left[1] + i * spacing
            cv2.rectangle(board, (ic_top_left[0] - 18, py - 3), (ic_top_left[0], py + 3), pin_color, -1)
            cv2.rectangle(board, (ic_bottom_right[0], py - 3), (ic_bottom_right[0] + 18, py + 3), pin_color, -1)

        # Secondary IC (Memory/Flash, top-right)
        m_top_left = (780, 180)
        m_bottom_right = (920, 320)
        cv2.rectangle(board, m_top_left, m_bottom_right, (40, 40, 40), -1)
        cv2.rectangle(board, m_top_left, m_bottom_right, (70, 70, 70), 2)
        cv2.putText(board, "SRAM 512M", (m_top_left[0] + 15, m_top_left[1] + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)

        # Memory chip pins (left & right)
        for i in range(1, 10):
            py = m_top_left[1] + i * 14
            cv2.rectangle(board, (m_top_left[0] - 14, py - 3), (m_top_left[0], py + 3), pin_color, -1)
            cv2.rectangle(board, (m_bottom_right[0], py - 3), (m_bottom_right[0] + 14, py + 3), pin_color, -1)

        # SMD Capacitors & Resistors (small surface-mount components)
        smd_locs = [
            (250, 200), (250, 240), (250, 280),
            (680, 420), (680, 460), (680, 500),
            (850, 550), (900, 550), (950, 550),
            (220, 600), (220, 650)
        ]
        for sx, sy in smd_locs:
            # Ceramic body (brown or black)
            cv2.rectangle(board, (sx - 15, sy - 8), (sx + 15, sy + 8), (20, 45, 90), -1)
            # Solder terminal caps
            cv2.rectangle(board, (sx - 15, sy - 8), (sx - 9, sy + 8), pin_color, -1)
            cv2.rectangle(board, (sx + 9, sy - 8), (sx + 15, sy + 8), pin_color, -1)

        # Test pads and fiducials (circular gold pads for optical calibration)
        fiducials = [(60, 60), (self.width - 60, 60), (60, self.height - 60), (self.width - 60, self.height - 60)]
        for fx, fy in fiducials:
            cv2.circle(board, (fx, fy), 24, (25, 80, 20), -1)
            cv2.circle(board, (fx, fy), 20, copper_color, 2)
            cv2.circle(board, (fx, fy), 10, copper_color, -1)
            cv2.circle(board, (fx, fy), 4, (25, 80, 20), -1)

        # Silk screen annotations
        cv2.putText(board, "REV 5.2 - AWS/OPENCV-COMP26", (self.width - 380, self.height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(board, "QC-PASS OPTISCAN-READY", (50, self.height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return board

    def generate_inspected_board(self, defect_type: str = "solder_bridge",
                                glare: bool = True,
                                noise_level: float = 0.02) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates an inspected board sample with realistic physical defects and optical noise.
        Defect types:
        - 'none' (healthy golden board with slight operational noise)
        - 'solder_bridge' (accidental short between IC pins)
        - 'hairline_crack' (trace discontinuity requiring active zoom & high contrast)
        - 'missing_component' (missing SMD capacitor leaving bare solder pads)
        - 'foreign_particle' (conductive dust/flux residue)
        """
        board = self.generate_golden_board()
        ground_truth = {
            "has_defect": defect_type != "none",
            "defect_type": defect_type,
            "target_roi": None,
            "description": ""
        }

        pin_color = (200, 205, 210)

        if defect_type == "solder_bridge":
            # Defect on top IC pins: solder bridge spanning pins 4 & 5
            ic_center = (450, 450)
            ic_top_left = (ic_center[0] - 80, ic_center[1] - 80)
            spacing = 160 // 13
            px1 = ic_top_left[0] + 4 * spacing
            px2 = ic_top_left[0] + 5 * spacing
            py = ic_top_left[1] - 9
            # Draw solder blob bridging the two pins
            cv2.ellipse(board, ((px1 + px2)//2, py), (spacing//2 + 5, 8), 0, 0, 360, (190, 195, 205), -1)
            cv2.circle(board, ((px1 + px2)//2, py), 4, (220, 225, 230), -1)
            roi_x, roi_y = int(px1 - 25), int(py - 25)
            ground_truth["target_roi"] = [roi_x, roi_y, 60, 50]
            ground_truth["description"] = "Micro solder bridge bridging IC-Pin-4 and Pin-5 (electrical short)"

        elif defect_type == "hairline_crack":
            # Hairline crack on high-current trace near (520, 300)
            cx, cy = 520, 300
            # Cut a 10px break across the copper trace with substrate color
            cv2.rectangle(board, (cx - 10, cy - 8), (cx + 10, cy + 8), (25, 80, 20), -1)
            # Add irregular dark fracture line
            cv2.line(board, (cx - 6, cy - 8), (cx + 6, cy + 8), (12, 28, 8), 2, cv2.LINE_AA)
            cv2.line(board, (cx - 4, cy + 2), (cx + 8, cy - 4), (12, 28, 8), 1, cv2.LINE_AA)
            ground_truth["target_roi"] = [cx - 30, cy - 30, 60, 60]
            ground_truth["description"] = "Sub-millimeter trace discontinuity / hairline thermal crack"

        elif defect_type == "missing_component":
            # Missing SMD at (680, 460)
            sx, sy = 680, 460
            # Paint over component body with bare pad flux marks
            cv2.rectangle(board, (sx - 15, sy - 8), (sx + 15, sy + 8), (28, 92, 23), -1)
            # Unsoldered exposed pads with dull flux
            cv2.rectangle(board, (sx - 13, sy - 7), (sx - 9, sy + 7), (120, 130, 135), -1)
            cv2.rectangle(board, (sx + 9, sy - 7), (sx + 13, sy + 7), (120, 130, 135), -1)
            ground_truth["target_roi"] = [sx - 35, sy - 25, 70, 50]
            ground_truth["description"] = "Missing SMD capacitor C46; unpopulated solder pads"

        elif defect_type == "foreign_particle":
            # Conductive flake bridging memory pins at (780, 210)
            fx, fy = 775, 212
            cv2.fillPoly(board, [np.array([[fx-8, fy-4], [fx+12, fy-2], [fx+5, fy+9], [fx-6, fy+5]], dtype=np.int32)], (160, 165, 175))
            ground_truth["target_roi"] = [fx - 25, fy - 25, 55, 55]
            ground_truth["description"] = "Foreign metallic debris across SRAM bus lines"

        else:
            # Clean board
            ground_truth["target_roi"] = [450, 450, 60, 60]
            ground_truth["description"] = "Nominal board state (within manufacturing tolerance)"

        # Introduce realistic industrial environmental lighting/glare
        if glare:
            # Specular highlight hotspot from overhead factory lighting
            glare_center = (random.randint(300, 800), random.randint(200, 600))
            y_indices, x_indices = np.indices((self.height, self.width))
            dist_sq = (x_indices - glare_center[0])**2 + (y_indices - glare_center[1])**2
            sigma = 220.0
            glare_map = np.exp(-dist_sq / (2 * sigma**2)) * 65.0
            glare_overlay = np.repeat(glare_map[:, :, np.newaxis], 3, axis=2).astype(np.float32)
            board_float = np.clip(board.astype(np.float32) + glare_overlay, 0, 255).astype(np.uint8)
            board = board_float

        # Introduce Gaussian sensor noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, board.shape).astype(np.float32)
            board = np.clip(board.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return board, ground_truth
