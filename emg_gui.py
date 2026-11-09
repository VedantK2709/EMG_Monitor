import csv
import math
import random
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import serial
from serial.tools import list_ports
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox,
    QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QVBoxLayout, QWidget)

BAUD = 115200
RATE_HZ = 100
WINDOW_SEC = 10

class Monitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EMG Muscle Sensor v3 | Live Monitor')
        self.resize(1100, 690)
        self.serial_port = None
        self.demo = False
        self.start = time.monotonic()
        self.buffer = bytearray()
        self.times = deque(maxlen=RATE_HZ * WINDOW_SEC * 2)
        self.volts = deque(maxlen=RATE_HZ * WINDOW_SEC * 2)
        self.records = []
        self.last_ms = None
        self.baseline = 0.1
        self.maximum = 1.5
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
        self.refresh = QPushButton('Refresh ports')
        self.refresh.clicked.connect(self.refresh_ports)
        self.connect_btn = QPushButton('Connect')
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.demo_btn = QPushButton('Demo mode')
        self.demo_btn.clicked.connect(self.toggle_demo)
        self.save_btn = QPushButton('Export CSV')
        self.save_btn.clicked.connect(self.save_csv)
        for widget in [QLabel('Serial port:'), self.ports, self.refresh,
                       self.connect_btn, self.demo_btn, self.save_btn]:
            top.addWidget(widget)
        layout.addLayout(top)
        self.status = QLabel('Disconnected | Connect ESP32 or start demo mode')
        layout.addWidget(self.status)

        self.graph = pg.PlotWidget(title='Live muscle activation envelope (ADC-pin voltage)')
        self.graph.setLabel('left', 'ADC-pin voltage', units='V')
        self.graph.setLabel('bottom', 'Time', units='s')
        self.graph.showGrid(x=True, y=True, alpha=0.25)
        self.graph.setYRange(0, 3.3)
        self.curve = self.graph.plot(pen=pg.mkPen('#00bfa5', width=2))
        layout.addWidget(self.graph, stretch=1)

        grid = QGridLayout()
        self.readout = {}
        for i, name in enumerate(['Voltage', 'ADC counts', 'Activation',
                                  'Peak (10s)', 'Mean (10s)', 'Muscle status', 'Samples']):
            grid.addWidget(QLabel(name + ':'), i // 4, (i % 4) * 2)
            value = QLabel('—')
            value.setStyleSheet('font-size: 18px; font-weight: bold;')
            grid.addWidget(value, i // 4, (i % 4) * 2 + 1)
            self.readout[name] = value
        layout.addLayout(grid)

        calibration = QHBoxLayout()
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(0, 3.3)
        self.baseline_spin.setSingleStep(0.05)
        self.baseline_spin.setValue(self.baseline)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0.01, 3.3)
        self.max_spin.setSingleStep(0.05)
        self.max_spin.setValue(self.maximum)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setSuffix(' %')
        self.threshold_spin.setValue(20)
        for label, spin in [('Rest voltage (V)', self.baseline_spin),
                            ('Strong contraction voltage (V)', self.max_spin),
                            ('Active threshold', self.threshold_spin)]:
            calibration.addWidget(QLabel(label))
            calibration.addWidget(spin)
        layout.addLayout(calibration)
        layout.addWidget(QLabel('Voltage is measured at the ESP32 ADC pin, not necessarily at the sensor SIG pin. '
                                'Activation % is relative to manually entered calibration points.'))

    def refresh_ports(self):
        previous = self.ports.currentText()
        self.ports.clear()
        self.ports.addItems([p.device for p in list_ports.comports()])
        if previous:
            index = self.ports.findText(previous)
            if index >= 0:
                self.ports.setCurrentIndex(index)

    def toggle_connection(self):
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            self.connect_btn.setText('Connect')
            self.status.setText('Disconnected')
            return
        port = self.ports.currentText()
        if not port:
            QMessageBox.warning(self, 'No serial port', 'Connect ESP32 via USB and refresh ports.')
            return
        try:
            self.serial_port = serial.Serial(port, BAUD, timeout=0)
            self.buffer.clear()
            self.demo = False
            self.demo_btn.setText('Demo mode')
            self.connect_btn.setText('Disconnect')
            self.status.setText(f'Connected: {port} @ {BAUD} baud')
        except serial.SerialException as exc:
            QMessageBox.critical(self, 'Serial connection error', str(exc))

    def toggle_demo(self):
        self.demo = not self.demo
        if self.demo and self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            self.connect_btn.setText('Connect')
        self.demo_btn.setText('Stop demo' if self.demo else 'Demo mode')
        self.status.setText('SIMULATED DATA — not a real sensor' if self.demo else 'Disconnected')

    def add_sample(self, t, adc, mv):
        v = mv / 1000.0
        self.times.append(t)
        self.volts.append(v)
        self.records.append((datetime.now().isoformat(timespec='milliseconds'), t, adc, mv))
        self.readout['Voltage'].setText(f'{v:.3f} V')
        self.readout['ADC counts'].setText(f'{adc} / 4095')
        lo = self.baseline_spin.value()
        hi = self.max_spin.value()
        pct = 100 * (v - lo) / (hi - lo) if hi > lo else 0
        pct = min(100, max(0, pct))
        self.readout['Activation'].setText(f'{pct:.1f} %')
        self.readout['Muscle status'].setText('ACTIVE' if pct >= self.threshold_spin.value() else 'RELAXED')
        self.readout['Samples'].setText(str(len(self.records)))

    def update_data(self):
        if self.demo:
            now = time.monotonic() - self.start
            for i in range(3):
                t = now + i / RATE_HZ
                contraction = max(0, math.sin(2 * math.pi * t / 4)) ** 3
                v = min(3.3, max(0, 0.12 + 1.65 * contraction + random.gauss(0, 0.035)))
                self.add_sample(t, int(v / 3.3 * 4095), int(v * 1000))
        elif self.serial_port:
            try:
                incoming = self.serial_port.read(self.serial_port.in_waiting or 1)
                self.buffer.extend(incoming)
                while b'\n' in self.buffer:
                    line, _, rest = self.buffer.partition(b'\n')
                    self.buffer = bytearray(rest)
                    try:
                        ms, adc, mv = map(int, line.decode('ascii').strip().split(','))
                        if 0 <= adc <= 4095 and 0 <= mv <= 3300:
                            if self.last_ms is not None and ms < self.last_ms:
                                self.times.clear()
                                self.volts.clear()
                            self.last_ms = ms
                            self.add_sample(ms / 1000.0, adc, mv)
                    except (ValueError, UnicodeDecodeError):
                        pass
                if len(self.buffer) > 8192:
                    self.buffer.clear()
            except (serial.SerialException, OSError) as exc:
                self.status.setText(f'Serial disconnected: {exc}')
                self.serial_port.close()
                self.serial_port = None
                self.connect_btn.setText('Connect')
        if self.times:
            t = np.asarray(self.times)
            y = np.asarray(self.volts)
            cutoff = t[-1] - WINDOW_SEC
            mask = t >= cutoff
            self.curve.setData(t[mask], y[mask])
            self.graph.setXRange(max(0, t[-1] - WINDOW_SEC), max(WINDOW_SEC, t[-1]), padding=0)
            self.readout['Peak (10s)'].setText(f'{np.max(y[mask]):.3f} V')
            self.readout['Mean (10s)'].setText(f'{np.mean(y[mask]):.3f} V')

    def save_csv(self):
        if not self.records:
            QMessageBox.information(self, 'No data', 'Collect some samples first.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export EMG CSV', 'emg_recording.csv', 'CSV (*.csv)')
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['computer_timestamp', 'esp32_time_s', 'adc_count', 'adc_pin_mV'])
                    writer.writerows(self.records)
                self.status.setText(f'Saved {len(self.records)} samples to {path}')
            except OSError as exc:
                QMessageBox.critical(self, 'Save error', str(exc))

    def closeEvent(self, event):
        if self.serial_port:
            self.serial_port.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Monitor()
    window.show()
    sys.exit(app.exec())
