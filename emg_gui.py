"""
EMG Muscle Sensor v3 - Real-time desktop monitor

Expected ESP32 serial format (one sample per line):
    elapsed_ms,adc_count,adc_pin_mV
Example:
    1250,1832,1478

The voltage is the measured voltage at the ESP32 ADC pin.
This application does not reconstruct raw EMG; the Muscle Sensor v3
manual describes an amplified, rectified and smoothed output.
"""

import os

# IMPORTANT: Select the Qt binding BEFORE importing pyqtgraph.
# This application consistently uses PyQt6 (not PySide6).
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt6"

import csv
import math
import random
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

import pyqtgraph as pg

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

BAUD = 115200
WINDOW_SEC = 10
ADC_MAX = 4095
ADC_VREF_MV = 3300


class Monitor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EMG Muscle Sensor v3 | Live Monitor")
        self.resize(1100, 720)

        self.serial_port = None
        self.demo = False
        self.demo_start = time.monotonic()
        self.serial_buffer = bytearray()

        max_points = 5000
        self.times = deque(maxlen=max_points)
        self.volts = deque(maxlen=max_points)
        self.records = []
        self.last_ms = None

        self.build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(30)

        self.refresh_ports()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.ports = QComboBox()

        self.refresh_btn = QPushButton("Refresh ports")
        self.refresh_btn.clicked.connect(self.refresh_ports)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.demo_btn = QPushButton("Demo mode")
        self.demo_btn.clicked.connect(self.toggle_demo)

        self.save_btn = QPushButton("Export CSV")
        self.save_btn.clicked.connect(self.save_csv)

        for widget in (
            QLabel("Serial port:"),
            self.ports,
            self.refresh_btn,
            self.connect_btn,
            self.demo_btn,
            self.save_btn,
        ):
            top.addWidget(widget)

        layout.addLayout(top)

        self.status = QLabel(
            "Disconnected | Connect ESP32 or start demo mode"
        )
        self.status.setStyleSheet("font-weight: bold; padding: 6px;")
        layout.addWidget(self.status)

        self.graph = pg.PlotWidget(
            title="Live muscle activation envelope (ADC-pin voltage)"
        )
        self.graph.setLabel("left", "ADC-pin voltage", units="V")
        self.graph.setLabel("bottom", "Time", units="s")
        self.graph.showGrid(x=True, y=True, alpha=0.25)
        self.graph.setYRange(0, 3.3)
        self.graph.setBackground("w")
        self.curve = self.graph.plot(
            pen=pg.mkPen("#00a88f", width=2)
        )
        layout.addWidget(self.graph, stretch=1)

        grid = QGridLayout()
        self.readout = {}

        names = [
            "Voltage",
            "ADC counts",
            "Activation",
            "Peak (10s)",
            "Mean (10s)",
            "Muscle status",
            "Samples",
        ]

        for i, name in enumerate(names):
            row = i // 4
            col = (i % 4) * 2

            grid.addWidget(QLabel(name + ":"), row, col)

            value = QLabel("—")
            value.setStyleSheet(
                "font-size: 18px; font-weight: bold; padding: 4px;"
            )
            grid.addWidget(value, row, col + 1)
            self.readout[name] = value

        layout.addLayout(grid)

        calibration = QHBoxLayout()

        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(0, 3.3)
        self.baseline_spin.setSingleStep(0.05)
        self.baseline_spin.setDecimals(3)
        self.baseline_spin.setValue(0.10)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0.001, 3.3)
        self.max_spin.setSingleStep(0.05)
        self.max_spin.setDecimals(3)
        self.max_spin.setValue(1.50)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setSuffix(" %")
        self.threshold_spin.setValue(20)

        for label, spin in (
            ("Rest voltage (V)", self.baseline_spin),
            ("Strong contraction voltage (V)", self.max_spin),
            ("Active threshold", self.threshold_spin),
        ):
            calibration.addWidget(QLabel(label))
            calibration.addWidget(spin)

        layout.addLayout(calibration)

        note = QLabel(
            "Voltage is measured at the ESP32 ADC pin. Activation % is a "
            "relative value based on the calibration points above; it is "
            "not a medical measurement."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

    def refresh_ports(self):
        previous = self.ports.currentText()
        self.ports.clear()

        devices = [port.device for port in list_ports.comports()]
        self.ports.addItems(devices)

        if previous:
            index = self.ports.findText(previous)
            if index >= 0:
                self.ports.setCurrentIndex(index)

        if not devices:
            self.status.setText(
                "No serial ports found | Connect ESP32 and refresh ports"
            )

    def toggle_connection(self):
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None
            self.connect_btn.setText("Connect")
            self.status.setText("Disconnected")
            return

        port = self.ports.currentText()
        if not port:
            QMessageBox.warning(
                self,
                "No serial port",
                "Connect the ESP32 using a USB data cable and refresh ports.",
            )
            return

        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=BAUD,
                timeout=0,
            )
            self.serial_buffer.clear()
            self.last_ms = None
            self.demo = False
            self.demo_btn.setText("Demo mode")
            self.connect_btn.setText("Disconnect")
            self.status.setText(f"Connected: {port} @ {BAUD} baud")

        except (serial.SerialException, OSError) as exc:
            self.serial_port = None
            QMessageBox.critical(
                self, "Serial connection error", str(exc)
            )

    def toggle_demo(self):
        self.demo = not self.demo

        if self.demo and self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None
            self.connect_btn.setText("Connect")

        if self.demo:
            self.demo_start = time.monotonic()
            self.demo_btn.setText("Stop demo")
            self.status.setText(
                "SIMULATED DATA — not readings from a real sensor"
            )
        else:
            self.demo_btn.setText("Demo mode")
            self.status.setText("Disconnected")

    def add_sample(self, t_seconds, adc_count, millivolts):
        voltage = millivolts / 1000.0

        self.times.append(t_seconds)
        self.volts.append(voltage)

        self.records.append(
            (
                datetime.now().isoformat(timespec="milliseconds"),
                t_seconds,
                adc_count,
                millivolts,
            )
        )

        self.readout["Voltage"].setText(f"{voltage:.3f} V")
        self.readout["ADC counts"].setText(
            f"{adc_count} / {ADC_MAX}"
        )

        baseline = self.baseline_spin.value()
        maximum = self.max_spin.value()

        if maximum > baseline:
            activation = 100.0 * (voltage - baseline) / (
                maximum - baseline
            )
        else:
            activation = 0.0

        activation = max(0.0, min(100.0, activation))

        self.readout["Activation"].setText(f"{activation:.1f} %")
        self.readout["Muscle status"].setText(
            "ACTIVE"
            if activation >= self.threshold_spin.value()
            else "RELAXED"
        )
        self.readout["Samples"].setText(str(len(self.records)))

    def read_serial_samples(self):
        if self.serial_port is None:
            return

        try:
            incoming = self.serial_port.read(
                self.serial_port.in_waiting or 1
            )
            if not incoming:
                return

            self.serial_buffer.extend(incoming)

            while b"\n" in self.serial_buffer:
                line, _, remainder = self.serial_buffer.partition(b"\n")
                self.serial_buffer = bytearray(remainder)

                try:
                    fields = line.decode("ascii").strip().split(",")
                    if len(fields) != 3:
                        continue

                    ms, adc, mv = map(int, fields)

                    if not (
                        0 <= adc <= ADC_MAX
                        and 0 <= mv <= ADC_VREF_MV
                        and ms >= 0
                    ):
                        continue

                    # ESP32 millis() rolls over after a long uptime.
                    if self.last_ms is not None and ms < self.last_ms:
                        self.times.clear()
                        self.volts.clear()

                    self.last_ms = ms
                    self.add_sample(ms / 1000.0, adc, mv)

                except (ValueError, UnicodeDecodeError):
                    # Ignore incomplete or non-data serial lines.
                    continue

            if len(self.serial_buffer) > 8192:
                self.serial_buffer.clear()

        except (serial.SerialException, OSError) as exc:
            self.status.setText(f"Serial disconnected: {exc}")
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None
            self.connect_btn.setText("Connect")

    def generate_demo_samples(self):
        now = time.monotonic() - self.demo_start

        # Three samples every 30 ms gives approximately 100 samples/s.
        for i in range(3):
            t = now + i / 100.0
            contraction = max(
                0.0, math.sin(2 * math.pi * t / 4.0)
            ) ** 3
            voltage = min(
                3.3,
                max(
                    0.0,
                    0.12
                    + 1.65 * contraction
                    + random.gauss(0, 0.035),
                ),
            )
            adc = round(voltage / 3.3 * ADC_MAX)
            mv = round(voltage * 1000)
            self.add_sample(t, adc, mv)

    def update_graph(self):
        if not self.times:
            return

        t = np.asarray(self.times, dtype=float)
        y = np.asarray(self.volts, dtype=float)

        cutoff = t[-1] - WINDOW_SEC
        mask = t >= cutoff

        visible_t = t[mask]
        visible_y = y[mask]

        if len(visible_t) == 0:
            return

        self.curve.setData(visible_t, visible_y)

        left = max(0.0, visible_t[-1] - WINDOW_SEC)
        right = max(WINDOW_SEC, visible_t[-1])
        self.graph.setXRange(left, right, padding=0)

        self.readout["Peak (10s)"].setText(
            f"{np.max(visible_y):.3f} V"
        )
        self.readout["Mean (10s)"].setText(
            f"{np.mean(visible_y):.3f} V"
        )

    def update_data(self):
        if self.demo:
            self.generate_demo_samples()
        elif self.serial_port is not None:
            self.read_serial_samples()

        self.update_graph()

    def save_csv(self):
        if not self.records:
            QMessageBox.information(
                self, "No data", "Collect some samples first."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export EMG CSV",
            "emg_recording.csv",
            "CSV files (*.csv)",
        )

        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "computer_timestamp",
                        "esp32_time_s",
                        "adc_count",
                        "adc_pin_mV",
                    ]
                )
                writer.writerows(self.records)

            self.status.setText(
                f"Saved {len(self.records)} samples to {path}"
            )

        except OSError as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def closeEvent(self, event):
        self.timer.stop()

        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Monitor()
    window.show()
    sys.exit(app.exec())
