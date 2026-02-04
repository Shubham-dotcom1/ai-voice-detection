import numpy as np
from typing import Dict, List
from .feature_extractor import AudioFeatureExtractor

class VoiceDetector:
    """
    AI vs Human Voice Detection Engine
    """
    
    THRESHOLDS = {
        'pitch_cv_ai_max': 0.08,
        'pitch_cv_human_min': 0.12,
        'pitch_std_ai_max': 15,
        'mfcc_variance_ai_max': 80,
        'mfcc_delta_std_ai_max': 10,
        'spectral_flatness_ai_min': 0.05,
        'spectral_centroid_std_ai_max': 200,
        'silence_ratio_ai_max': 0.03,
        'silence_ratio_ai_min': 0.35,
        'jitter_ai_max': 0.02,
        'harmonic_ratio_ai_min': 50,
    }
    
    WEIGHTS = {
        'pitch': 0.25,
        'mfcc': 0.20,
        'spectral': 0.20,
        'temporal': 0.15,
        'voice_quality': 0.20
    }
    
    def __init__(self, y: np.ndarray, sr: int, language: str):
        self.y = y
        self.sr = sr
        self.language = language
        self.features: Dict[str, float] = {}
        self.scores: Dict[str, float] = {}
        self.indicators: List[str] = []
    
    def detect(self) -> Dict:
        """Main detection method"""
        extractor = AudioFeatureExtractor(self.y, self.sr)
        self.features = extractor.extract_all_features()
        
        self._analyze_pitch()
        self._analyze_mfcc()
        self._analyze_spectral()
        self._analyze_temporal()
        self._analyze_voice_quality()
        
        ai_probability = self._calculate_final_score()
        
        if ai_probability >= 0.55:
            classification = "AI_GENERATED"
            confidence = ai_probability
        else:
            classification = "HUMAN"
            confidence = 1.0 - ai_probability
        
        confidence = max(0.51, min(0.99, confidence))
        explanation = self._generate_explanation(classification)
        
        return {
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }
    
    def _analyze_pitch(self):
        """Analyze pitch features"""
        score = 0.5
        pitch_cv = self.features.get('pitch_cv', 0)
        pitch_std = self.features.get('pitch_std', 0)
        
        if pitch_cv < self.THRESHOLDS['pitch_cv_ai_max']:
            score = 0.85
            self.indicators.append("Unnaturally consistent pitch detected")
        elif pitch_cv < self.THRESHOLDS['pitch_cv_human_min']:
            score = 0.65
            self.indicators.append("Limited pitch variation observed")
        else:
            score = 0.25
            self.indicators.append("Natural pitch variation present")
        
        if pitch_std < self.THRESHOLDS['pitch_std_ai_max'] and pitch_std > 0:
            score = min(score + 0.1, 0.95)
        
        self.scores['pitch'] = score
    
    def _analyze_mfcc(self):
        """Analyze MFCC features"""
        score = 0.5
        mfcc_var = self.features.get('mfcc_variance', 0)
        mfcc_delta_std = self.features.get('mfcc_delta_std', 0)
        
        if mfcc_var < self.THRESHOLDS['mfcc_variance_ai_max']:
            score = 0.75
            self.indicators.append("Low spectral complexity detected")
        else:
            score = 0.35
        
        if mfcc_delta_std < self.THRESHOLDS['mfcc_delta_std_ai_max']:
            score = min(score + 0.15, 0.9)
            self.indicators.append("Limited dynamic speech patterns")
        
        self.scores['mfcc'] = score
    
    def _analyze_spectral(self):
        """Analyze spectral features"""
        score = 0.5
        flatness = self.features.get('spectral_flatness_mean', 0)
        centroid_std = self.features.get('spectral_centroid_std', 0)
        
        if flatness > self.THRESHOLDS['spectral_flatness_ai_min']:
            score = 0.7
            self.indicators.append("Unusual spectral characteristics")
        else:
            score = 0.4
        
        if centroid_std < self.THRESHOLDS['spectral_centroid_std_ai_max']:
            score = min(score + 0.1, 0.85)
        
        self.scores['spectral'] = score
    
    def _analyze_temporal(self):
        """Analyze temporal features"""
        score = 0.5
        silence_ratio = self.features.get('silence_ratio', 0)
        
        if silence_ratio < self.THRESHOLDS['silence_ratio_ai_max']:
            score = 0.8
            self.indicators.append("Lack of natural breathing pauses")
        elif silence_ratio > self.THRESHOLDS['silence_ratio_ai_min']:
            score = 0.65
            self.indicators.append("Unusual pause patterns detected")
        else:
            score = 0.3
            self.indicators.append("Natural speech rhythm detected")
        
        self.scores['temporal'] = score
    
    def _analyze_voice_quality(self):
        """Analyze voice quality"""
        score = 0.5
        jitter = self.features.get('jitter', 0)
        harmonic_ratio = self.features.get('harmonic_ratio', 0)
        
        if jitter < self.THRESHOLDS['jitter_ai_max']:
            score = 0.8
            self.indicators.append("Missing natural voice micro-variations")
        else:
            score = 0.3
            self.indicators.append("Natural voice tremor patterns present")
        
        if harmonic_ratio > self.THRESHOLDS['harmonic_ratio_ai_min']:
            score = min(score + 0.1, 0.9)
            self.indicators.append("Unusually clean audio signal")
        
        self.scores['voice_quality'] = score
    
    def _calculate_final_score(self) -> float:
        """Calculate weighted final score"""
        total_score = 0.0
        total_weight = 0.0
        
        for category, weight in self.WEIGHTS.items():
            if category in self.scores:
                total_score += self.scores[category] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.5
    
    def _generate_explanation(self, classification: str) -> str:
        """Generate explanation"""
        if classification == "AI_GENERATED":
            ai_indicators = [
                ind for ind in self.indicators 
                if any(word in ind.lower() for word in [
                    'unnatural', 'lack', 'missing', 'unusual', 
                    'limited', 'low', 'synthetic', 'clean'
                ])
            ]
            
            if ai_indicators:
                return ai_indicators[0]
            
            max_score_category = max(self.scores, key=self.scores.get)
            
            explanations = {
                'pitch': "Unnatural pitch consistency and robotic speech patterns detected",
                'mfcc': "Synthetic spectral patterns identified in voice analysis",
                'spectral': "Artificial frequency distribution detected",
                'temporal': "Mechanical timing patterns without natural rhythm",
                'voice_quality': "Missing natural voice micro-variations and tremors"
            }
            
            return explanations.get(max_score_category, 
                "Synthetic speech patterns detected in audio analysis")
        
        else:
            human_indicators = [
                ind for ind in self.indicators 
                if any(word in ind.lower() for word in [
                    'natural', 'present', 'human', 'normal'
                ])
            ]
            
            if human_indicators:
                return human_indicators[0]
            
            min_score_category = min(self.scores, key=self.scores.get)
            
            explanations = {
                'pitch': "Natural pitch variation consistent with human speech",
                'mfcc': "Complex spectral patterns typical of human voice",
                'spectral': "Natural frequency characteristics detected",
                'temporal': "Organic speech rhythm with natural pauses",
                'voice_quality': "Natural voice micro-variations and breathing detected"
            }
            
            return explanations.get(min_score_category,
                "Natural human speech characteristics identified")