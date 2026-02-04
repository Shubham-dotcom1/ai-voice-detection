import numpy as np
import librosa
from typing import Dict

class AudioFeatureExtractor:
    """
    Extract audio features for AI voice detection
    """
    
    def __init__(self, y: np.ndarray, sr: int = 16000):
        self.y = y
        self.sr = sr
        self.features: Dict[str, float] = {}
    
    def extract_all_features(self) -> Dict[str, float]:
        """Extract all audio features"""
        
        self._extract_mfcc_features()
        self._extract_pitch_features()
        self._extract_spectral_features()
        self._extract_temporal_features()
        self._extract_voice_quality_features()
        
        return self.features
    
    def _extract_mfcc_features(self):
        """MFCC features - AI voices often have uniform patterns"""
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        
        self.features['mfcc_variance'] = float(np.var(mfccs))
        self.features['mfcc_std_mean'] = float(np.mean(np.std(mfccs, axis=1)))
        
        mfcc_delta = librosa.feature.delta(mfccs)
        self.features['mfcc_delta_std'] = float(np.std(mfcc_delta))
        
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        self.features['mfcc_delta2_std'] = float(np.std(mfcc_delta2))
    
    def _extract_pitch_features(self):
        """Pitch analysis - AI voices have unnatural consistency"""
        pitches, magnitudes = librosa.piptrack(
            y=self.y, 
            sr=self.sr,
            fmin=50,
            fmax=500
        )
        
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 10:
            pitch_array = np.array(pitch_values)
            
            self.features['pitch_mean'] = float(np.mean(pitch_array))
            self.features['pitch_std'] = float(np.std(pitch_array))
            self.features['pitch_range'] = float(np.ptp(pitch_array))
            
            mean_pitch = np.mean(pitch_array)
            if mean_pitch > 0:
                self.features['pitch_cv'] = float(np.std(pitch_array) / mean_pitch)
            else:
                self.features['pitch_cv'] = 0.0
            
            pitch_diff = np.abs(np.diff(pitch_array))
            self.features['pitch_jumps'] = float(np.mean(pitch_diff > 50))
        else:
            self.features['pitch_mean'] = 0.0
            self.features['pitch_std'] = 0.0
            self.features['pitch_range'] = 0.0
            self.features['pitch_cv'] = 0.0
            self.features['pitch_jumps'] = 0.0
    
    def _extract_spectral_features(self):
        """Spectral features"""
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        self.features['spectral_centroid_mean'] = float(np.mean(cent))
        self.features['spectral_centroid_std'] = float(np.std(cent))
        
        bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
        self.features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
        self.features['spectral_bandwidth_std'] = float(np.std(bandwidth))
        
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        self.features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        
        flatness = librosa.feature.spectral_flatness(y=self.y)[0]
        self.features['spectral_flatness_mean'] = float(np.mean(flatness))
        self.features['spectral_flatness_std'] = float(np.std(flatness))
        
        contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        self.features['spectral_contrast_mean'] = float(np.mean(contrast))
    
    def _extract_temporal_features(self):
        """Temporal features - rhythm and pauses"""
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        self.features['zcr_mean'] = float(np.mean(zcr))
        self.features['zcr_std'] = float(np.std(zcr))
        
        rms = librosa.feature.rms(y=self.y)[0]
        self.features['rms_mean'] = float(np.mean(rms))
        self.features['rms_std'] = float(np.std(rms))
        
        if np.mean(rms) > 0:
            self.features['rms_cv'] = float(np.std(rms) / np.mean(rms))
        else:
            self.features['rms_cv'] = 0.0
        
        silence_threshold = 0.02 * np.max(np.abs(self.y))
        silent_frames = np.abs(self.y) < silence_threshold
        self.features['silence_ratio'] = float(np.mean(silent_frames))
        
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.features['tempo'] = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]) if len(tempo) > 0 else 0.0
    
    def _extract_voice_quality_features(self):
        """Voice quality - naturalness indicators"""
        harmonic = librosa.effects.harmonic(self.y)
        percussive = librosa.effects.percussive(self.y)
        
        h_energy = np.sum(harmonic ** 2)
        p_energy = np.sum(percussive ** 2)
        
        if p_energy > 0:
            self.features['harmonic_ratio'] = float(min(h_energy / p_energy, 100))
        else:
            self.features['harmonic_ratio'] = 100.0
        
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.010 * self.sr)
        
        if len(self.y) > frame_length:
            frames = librosa.util.frame(
                self.y, 
                frame_length=frame_length, 
                hop_length=hop_length
            )
            frame_energies = np.sum(frames ** 2, axis=0)
            
            if len(frame_energies) > 1 and np.mean(frame_energies) > 0:
                jitter = np.mean(np.abs(np.diff(frame_energies))) / np.mean(frame_energies)
                self.features['jitter'] = float(min(jitter, 1.0))
            else:
                self.features['jitter'] = 0.0
        else:
            self.features['jitter'] = 0.0
        
        self.features['shimmer'] = self.features.get('rms_cv', 0.0)