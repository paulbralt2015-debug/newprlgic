"""Décodeur Dolby Pro Logic (matrice active), d'après les brevets EP0630168A1 et US7386132B2.
Constantes de temps, retard et bande passante du surround : valeurs raisonnables,
pas des valeurs publiées par Dolby. Le réducteur de bruit Dolby B n'est pas implémenté."""
import numpy as np
from scipy.signal import butter, lfilter, sosfilt


def G(x):
    """G(x) = 1 - tan(45 - x), x en degrés de 0 à 45 (brevet US7386132B2, éq. 6)."""
    return 1.0 - np.tan(np.radians(45.0 - np.clip(x, 0.0, 45.0)))


class ProLogicDecoder:
    def __init__(self, fs=48000, env_ms=30.0, delay_ms=20.0, band=(100.0, 7000.0), headroom=0.6):
        a = float(np.exp(-1.0 / (fs * env_ms / 1000.0)))
        self.b, self.a = [1.0 - a], [1.0, -a]
        self.zi = {k: np.zeros(1) for k in "lrsd"}
        self.sos = butter(2, band, "bandpass", fs=fs, output="sos")
        self.bz = np.zeros((self.sos.shape[0], 2))
        self.buf = np.zeros(int(fs * delay_ms / 1000.0))
        self.h = headroom

    def _env(self, k, x):
        y, self.zi[k] = lfilter(self.b, self.a, np.abs(x), zi=self.zi[k])
        return y

    def process(self, x):
        """x : tableau (N, 2) Lt/Rt -> (N, 6) : FL, FR, FC, LFE, SL, SR."""
        L, R = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
        eL, eR = self._env("l", L), self._env("r", R)
        eS, eD = self._env("s", L + R), self._env("d", L - R)
        # angles de pilotage : lr > 0 = gauche, cs > 0 = avant/centre, cs < 0 = surround
        lr = np.degrees(np.arctan2(eL, eR)) - 45.0
        cs = np.degrees(np.arctan2(eS, eD)) - 45.0
        k = np.maximum(1.0, (np.abs(lr) + np.abs(cs)) / 45.0)  # |lr| + |cs| <= 45
        lr, cs = lr / k, cs / k
        quiet = (eL + eR) < 1e-4
        lr[quiet] = 0.0
        cs[quiet] = 0.0
        l, r = np.maximum(lr, 0.0), np.maximum(-lr, 0.0)
        cf, gc, sg = np.maximum(cs, 0.0), G(np.abs(cs)), np.sign(cs)
        # éléments de la matrice (brevet '89 / Pro Logic : éq. 3, 14 et 24)
        FL = (1 - 0.5 * gc + 0.41 * G(l)) * L - sg * 0.5 * gc * R
        FR = (1 - 0.5 * gc + 0.41 * G(r)) * R - sg * 0.5 * gc * L
        C = (1 - G(l) + 0.41 * G(cf)) * L + (1 - G(r) + 0.41 * G(cf)) * R
        S = (1 - G(l)) * L - (1 - G(r)) * R
        S, self.bz = sosfilt(self.sos, S, zi=self.bz)
        d = np.concatenate([self.buf, S])
        S, self.buf = d[:len(S)], d[len(S):]
        h = self.h
        out = np.zeros((len(L), 6), np.float32)
        out[:, 0], out[:, 1], out[:, 2] = FL * h, FR * h, C * 0.71 * h
        out[:, 4] = out[:, 5] = S * 0.71 * h
        return np.clip(out, -1.0, 1.0)
