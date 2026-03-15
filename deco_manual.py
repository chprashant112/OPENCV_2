import sys, cv2, numpy as np
import os, serial, serial.tools.list_ports
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore

# ================= SETTINGS =================
DELTA_E_THRESHOLD = 12.0  # Adjustable based on factory lighting
CLASS_MAP = {0: "ROI", 1: "COLOR", 2: "DECO"} 

def extract_vibrant_lab(roi):
    """
    UPGRADED STAGE 2: Adaptive Color Extraction.
    If vibrant pixels aren't found, it relaxes the filter to find 
    the dominant pigment while still avoiding pure black/white.
    """
    if roi is None or roi.size == 0: return None
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 1. ATTEMPT: Look for 'Vibrant' pixels (Saturation > 40, Value 50-245)
    mask = cv2.inRange(hsv, (0, 40, 50), (180, 255, 245))
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    vibrant_pixels = lab[mask > 0]
    
    if len(vibrant_pixels) > 20:
        return np.median(vibrant_pixels, axis=0)

    # 2. FALLBACK: If no vibrant pixels (thin boxes/shadows), relax the saturation filter
    # This helps when the DECO part is very dark or metallic
    relaxed_mask = cv2.inRange(hsv, (0, 10, 30), (180, 255, 250))
    relaxed_pixels = lab[relaxed_mask > 0]
    
    if len(relaxed_pixels) > 5:
        return np.median(relaxed_pixels, axis=0)
        
    # 3. LAST RESORT: Just return the median of the center of the box
    return np.median(lab.reshape(-1, 3), axis=0)
class DecoInspectionSystem(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DECO Color Inspection — SPEN Compact")
        self.setFixedSize(1300, 850)
        self.dataset_path = ""
        self.build_ui()

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # ---- LEFT: MAIN SCREEN & LOGS ----
        left_layout = QtWidgets.QVBoxLayout()
        
        self.screen = QtWidgets.QLabel()
        self.screen.setStyleSheet("background:black; border:3px solid #111; border-radius:5px;")
        self.screen.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.screen, 5)

        self.status_banner = QtWidgets.QLabel("WAITING")
        self.status_banner.setFixedHeight(80)
        self.status_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.status_banner.setStyleSheet("font-size:38px; font-weight:bold; border:4px solid black; background:white; color:#333;")
        left_layout.addWidget(self.status_banner)

        self.log_table = QtWidgets.QTableWidget(0, 4)
        self.log_table.setHorizontalHeaderLabels(["#", "Time", "Result", "ΔE"])
        self.log_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.log_table.setFixedHeight(180)
        left_layout.addWidget(self.log_table)
        
        main_layout.addLayout(left_layout, 4)

        # ---- RIGHT: CONTROLS (The Blue Box Area) ----
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setContentsMargins(10, 0, 10, 0)

        # Production Controls
        right_panel.addWidget(QtWidgets.QLabel("<b>Production Controls</b>"))
        self.btn_load = QtWidgets.QPushButton("Load YOLO Model")
        self.btn_start = QtWidgets.QPushButton("Start Live Feed")
        self.btn_start.setStyleSheet("background:#4CAF50; color:white; font-weight:bold;")
        self.btn_stop = QtWidgets.QPushButton("Stop Feed")
        self.btn_stop.setStyleSheet("background:#E91E63; color:white; font-weight:bold;")
        self.btn_manual_trig = QtWidgets.QPushButton("Manual Trigger")
        self.btn_manual_trig.setStyleSheet("background:#FF9800; color:white; font-weight:bold;")

        for btn in [self.btn_load, self.btn_start, self.btn_stop, self.btn_manual_trig]:
            btn.setFixedHeight(35)
            right_panel.addWidget(btn)

        # Hardware Config
        right_panel.addSpacing(10)
        right_panel.addWidget(QtWidgets.QLabel("<b>Arduino Hardware</b>"))
        self.combo_trig = QtWidgets.QComboBox()
        self.combo_out = QtWidgets.QComboBox()
        right_panel.addWidget(self.combo_trig)
        right_panel.addWidget(self.combo_out)

        # --- MANUAL TESTING FEED (INTEGRATED VALIDATOR) ---
        right_panel.addSpacing(20)
        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine); line.setFrameShadow(QtWidgets.QFrame.Sunken)
        right_panel.addWidget(line)
        
        right_panel.addWidget(QtWidgets.QLabel("<b>Manual Logic Testing</b>"))
        self.btn_browse = QtWidgets.QPushButton("📁 Browse Labeled Folder")
        self.btn_browse.clicked.connect(self.load_dataset_folder)
        right_panel.addWidget(self.btn_browse)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setStyleSheet("background:#f0f0f0; border:1px solid #999; font-size:11px;")
        self.file_list.itemClicked.connect(self.validate_labeled_file)
        right_panel.addWidget(self.file_list)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

        # Populate Ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.combo_trig.addItems(ports)
        self.combo_out.addItems(ports)

    # ================= VALIDATION LOGIC =================
    def load_dataset_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Labeled Dataset (Images + TXT)")
        if path:
            self.dataset_path = path
            self.file_list.clear()
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.file_list.addItems(sorted(images))

    def validate_labeled_file(self, item):
        filename = item.text()
        img_path = os.path.join(self.dataset_path, filename)
        lbl_path = os.path.join(self.dataset_path, filename.rsplit('.', 1)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None: return

        h, w, _ = img.shape
        display_frame = img.copy()
        body_lab, deco_lab = None, None

        # 1. Parse Labeled Areas
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    if len(parts) < 5: continue
                    cls, x, y, nw, nh = map(float, parts)
                    x1, y1 = int((x - nw/2) * w), int((y - nh/2) * h)
                    x2, y2 = int((x + nw/2) * w), int((y + nh/2) * h)
                    
                    roi = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                    lab_color = extract_vibrant_lab(roi)
                    
                    name = CLASS_MAP.get(int(cls), "Target")
                    box_color = (255, 120, 0) if "DECO" not in name else (0, 200, 255)
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(display_frame, name, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

                    if "ROI" in name or "COLOR" in name: body_lab = lab_color
                    if "DECO" in name: deco_lab = lab_color

        # 2. Compare Colors (Stage 2)
        if body_lab is not None and deco_lab is not None:
            de = np.linalg.norm(body_lab - deco_lab)
            # Similarity scaled for presentation: 0-100%
            similarity = max(0, 100 - (de * 3.5)) 
            is_match = de < DELTA_E_THRESHOLD
            
            status = "PASS" if is_match else "FAIL"
            bg_color = "#2e7d32" if is_match else "#c62828"
            
            self.status_banner.setText(f"{status} | Similarity: {similarity:.1f}%")
            self.status_banner.setStyleSheet(f"font-size:38px; font-weight:bold; border:4px solid black; background:{bg_color}; color:white;")
            self.add_log_entry(status, de)
        else:
            self.status_banner.setText("INCOMPLETE DATA")
            self.status_banner.setStyleSheet("font-size:38px; font-weight:bold; border:4px solid black; background:#444; color:white;")

        self.update_screen(display_frame)

    def add_log_entry(self, result, de):
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        self.log_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
        self.log_table.setItem(row, 1, QtWidgets.QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.log_table.setItem(row, 2, QtWidgets.QTableWidgetItem(result))
        self.log_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{de:.2f}"))
        self.log_table.scrollToBottom()

    def update_screen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.screen.setPixmap(pix.scaled(self.screen.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DecoInspectionSystem()
    win.show()
    sys.exit(app.exec_())