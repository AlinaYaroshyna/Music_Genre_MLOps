import librosa
import numpy as np


class FeatureExtractor:

    def __init__(
        self,
        sample_rate=22050,
        duration=30,
        n_mfcc=13
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc

    def load_audio(self, path):
        try:
            y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return y, sr
        except Exception as e:
            print(f"[AUDIO LOAD ERROR] {path} -> {e}")
            return None, None

    def extract_mfcc(self, y, sr):

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc
        )

        return np.mean(mfcc.T, axis=0).flatten()

    def extract_chroma(self, y, sr):

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr
        )

        return np.mean(chroma.T, axis=0).flatten()

    def extract_centroid(self, y, sr):
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
        return np.array([np.mean(centroid)]).flatten()

    def extract_zcr(self, y):
        zcr = librosa.feature.zero_crossing_rate(y)
        return np.array([np.mean(zcr)]).flatten()

    def extract_all(self, path):

        y, sr = self.load_audio(path)
        if y is None:
            return None
        mfcc = self.extract_mfcc(y, sr)
        chroma = self.extract_chroma(y, sr)
        centroid = self.extract_centroid(y, sr)
        zcr = self.extract_zcr(y)
        mfcc = np.array(mfcc).reshape(-1)
        chroma = np.array(chroma).reshape(-1)
        centroid = np.array([centroid]).reshape(-1)
        zcr = np.array([zcr]).reshape(-1)

        combined = np.concatenate([
            mfcc,
            chroma,
            centroid,
            zcr
        ])

        return {
            "mfcc": mfcc,
            "chroma": chroma,
            "centroid": centroid,
            "zcr": zcr,
            "combined": combined
        }

