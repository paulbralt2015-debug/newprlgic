import os, sys, locale, subprocess, tempfile
from pathlib import Path

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
os.environ["PATH"] = str(BASE) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(BASE))

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget)

locale.setlocale(locale.LC_NUMERIC, "C")
import mpv
from prologic import ProLogicDecoder

FS = 48000
FFMPEG = str(BASE / "ffmpeg.exe") if (BASE / "ffmpeg.exe").exists() else "ffmpeg"
NOWIN = 0x08000000 if os.name == "nt" else 0


class Worker(QThread):
    progress, done, failed = Signal(int), Signal(str), Signal(str)

    def __init__(self, path, aidx, dur):
        super().__init__()
        self.path, self.aidx, self.dur = path, aidx, dur

    def run(self):
        try:
            out = str(Path(tempfile.gettempdir()) / f"prologic_{abs(hash((self.path, self.aidx)))}.mka")
            dec = subprocess.Popen(
                [FFMPEG, "-v", "error", "-i", self.path, "-map", f"0:a:{self.aidx}", "-vn",
                 "-ac", "2", "-ar", str(FS), "-f", "f32le", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NOWIN)
            enc = subprocess.Popen(
                [FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(FS), "-ac", "6", "-i", "-",
                 "-c:a", "flac", "-sample_fmt", "s32", out],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NOWIN)
            pl, n = ProLogicDecoder(FS), 0
            while True:
                raw = dec.stdout.read(FS * 8)
                if not raw:
                    break
                raw = raw[: len(raw) // 8 * 8]
                x = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
                enc.stdin.write(pl.process(x).astype("<f4").tobytes())
                n += len(x)
                if self.dur:
                    self.progress.emit(min(99, int(n / FS / self.dur * 100)))
            enc.stdin.close()
            enc.wait()
            dec.wait()
            if enc.returncode or n == 0:
                raise RuntimeError("ffmpeg a échoué (piste audio illisible ?)")
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class Win(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lecteur Dolby Pro Logic")
        self.resize(1000, 640)
        self.path = self.pl_path = None
        self.orig_aid = self.pl_aid = None
        self.aidx = 0
        self.video = QWidget()
        self.video.setStyleSheet("background:black")
        self.video.setAttribute(Qt.WA_NativeWindow)
        self.p = mpv.MPV(wid=str(int(self.video.winId())), input_default_bindings=False,
                         keep_open="yes", ytdl=False)
        bo, bp = QPushButton("Ouvrir…"), QPushButton("Lecture / Pause")
        self.chk = QCheckBox("Décodage Dolby Pro Logic")
        self.bar, self.lbl = QProgressBar(), QLabel("")
        self.bar.setVisible(False)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(80)
        self.vol.setMaximumWidth(120)
        row = QHBoxLayout()
        for w in (bo, bp, self.chk, self.bar, self.lbl, QLabel("Vol."), self.vol):
            row.addWidget(w)
        lay = QVBoxLayout(self)
        lay.addWidget(self.video, 1)
        lay.addWidget(self.seek)
        lay.addLayout(row)
        bo.clicked.connect(self.open_file)
        bp.clicked.connect(lambda: setattr(self.p, "pause", not self.p.pause))
        self.chk.toggled.connect(self.toggle)
        self.seek.sliderMoved.connect(self.do_seek)
        self.vol.valueChanged.connect(lambda v: setattr(self.p, "volume", v))
        self.p.volume = 80
        t = QTimer(self)
        t.timeout.connect(self.tick)
        t.start(250)

    def open_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Ouvrir une vidéo")
        if not f:
            return
        self.path, self.pl_path = f, None
        self.chk.blockSignals(True)
        self.chk.setChecked(False)
        self.chk.blockSignals(False)
        self.p.play(f)
        self.p.wait_until_playing()
        tr = [t for t in self.p.track_list if t["type"] == "audio"]
        sel = [i for i, t in enumerate(tr) if t.get("selected")]
        self.aidx = sel[0] if sel else 0
        self.orig_aid = tr[self.aidx]["id"] if tr else None

    def toggle(self, on):
        if not self.path:
            self.chk.setChecked(False)
            return
        if not on:
            self.p.aid = self.orig_aid
        elif self.pl_path:
            self.p.aid = self.pl_aid
        else:
            self.chk.setEnabled(False)
            self.bar.setVisible(True)
            self.lbl.setText("Décodage…")
            self.w = Worker(self.path, self.aidx, self.p.duration or 0)
            self.w.progress.connect(self.bar.setValue)
            self.w.done.connect(self.ready)
            self.w.failed.connect(self.fail)
            self.w.start()

    def ready(self, out):
        self.pl_path = out
        self.p.command("audio-add", out, "select")
        ext = [t["id"] for t in self.p.track_list if t["type"] == "audio" and t.get("external")]
        self.pl_aid = ext[-1] if ext else None
        self.bar.setVisible(False)
        self.lbl.setText("Pro Logic actif (5.1)")
        self.chk.setEnabled(True)

    def fail(self, msg):
        self.bar.setVisible(False)
        self.lbl.setText("Erreur : " + msg)
        self.chk.setEnabled(True)
        self.chk.blockSignals(True)
        self.chk.setChecked(False)
        self.chk.blockSignals(False)

    def do_seek(self, v):
        if self.p.duration:
            self.p.seek(v / 1000 * self.p.duration, "absolute")

    def tick(self):
        d, t = self.p.duration, self.p.time_pos
        if d and t is not None and not self.seek.isSliderDown():
            self.seek.setValue(int(t / d * 1000))

    def closeEvent(self, e):
        self.p.terminate()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Win()
    w.show()
    sys.exit(app.exec())
