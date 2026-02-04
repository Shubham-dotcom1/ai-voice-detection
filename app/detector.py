import numpy as np
from typing import Dict, List
from .feature_extractor import AudioFeatureExtractor

class VoiceDetector:
    """
    Advanced AI vs Human Voice Detection Engine
    Improved accuracy with better thresholds and multi-feature analysis
    """
    
    # Fine-tuned thresholds for better accuracy
    THRESHOLDS = {
        # Pitch features (AI has very consistent pitch)
        'pitch_cv_ai_max': 0.15,
        'pitch_cv_human_min': 0.08,
        'pitch_std_ai_max': 25,
        
        # MFCC features (AI has more uniform patterns)
        'mfcc_variance_ai_max': 150,
        'mfcc_delta_std_ai_max': 15,
        
        # Spectral features
        'spectral_flatness_ai_min': 0.03,
        'spectral_centroid_std_ai_max': 350,
        
        # Temporal features (AI lacks natural pauses)
        'silence_ratio_human_min': 0.05,
        'silence_ratio_human_max': 0.30,
        
        # Voice quality (AI lacks micro-variations)
        'jitter_ai_max': 0.05,
        'shimmer_ai_max': 0.3,
        'harmonic_ratio_ai_min': 40,
    }
    
    # Weights for each feature category
    WEIGHTS = {
        'pitch': 0.30,
        'mfcc': 0.20,
        'spectral': 0.15,
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
        self.ai_signals = 0
        self.human_signals = 0
    
    def detect(self) -> Dict:
        """Main detection method with improved accuracy"""
        
        # Extract features
        extractor = AudioFeatureExtractor(self.y, self.sr)
        self.features = extractor.extract_all_features()
        
        # Analyze each feature category
        self._analyze_pitch()
        self._analyze_mfcc()
        self._analyze_spectral()
        self._analyze_temporal()
        self._analyze_voice_quality()
        
        # Calculate final probability
        ai_probability = self._calculate_final_score()
        
        # Boost confidence based on signal count
        confidence_boost = self._calculate_confidence_boost()
        
        # Determine classification
        if ai_probability >= 0.50:
            classification = "AI_GENERATED"
            raw_confidence = ai_probability
        else:
            classification = "HUMAN"
            raw_confidence = 1.0 - ai_probability
        
        # Apply confidence boost
        final_confidence = min(0.99, raw_confidence + confidence_boost)
        final_confidence = max(0.55, final_confidence)
        
        # Generate explanation
        explanation = self._generate_explanation(classification)
        
        return {
            "classification": classification,
            "confidenceScore": round(final_confidence, 2),
            "explanation": explanation
        }
    
    def _analyze_pitch(self):
        """Analyze pitch features - key indicator for AI detection"""
        pitch_cv = self.features.get('pitch_cv', 0)
        pitch_std = self.features.get('pitch_std', 0)
        pitch_range = self.features.get('pitch_range', 0)
        
        ai_score = 0.0
        human_score = 0.0
        
        # Check pitch coefficient of variation
        if pitch_cv < self.THRESHOLDS['pitch_cv_human_min']:
            ai_score += 0.4
            self.ai_signals += 1
            self.indicators.append("Unnaturally consistent pitch detected")
        elif pitch_cv > self.THRESHOLDS['pitch_cv_ai_max']:
            human_score += 0.4
            self.human_signals += 1
            self.indicators.append("Natural pitch variation present")
        else:
            human_score += 0.2
        
        # Check pitch standard deviation
        if pitch_std < self.THRESHOLDS['pitch_std_ai_max'] and pitch_std > 0:
            ai_score += 0.3
            self.ai_signals += 1
        else:
            human_score += 0.3
            self.human_signals += 1
        
        # Check pitch range
        if pitch_range > 50:
            human_score += 0.3
            self.human_signals += 1
            self.indicators.append("Wide pitch range typical of human speech")
        else:
            ai_score += 0.2
        
        total = ai_score + human_score
        if total > 0:
            self.scores['pitch'] = ai_score / total
        else:
            self.scores['pitch'] = 0.5
    
    def _analyze_mfcc(self):
        """Analyze MFCC features"""
        mfcc_var = self.features.get('mfcc_variance', 0)
        mfcc_delta_std = self.features.get('mfcc_delta_std', 0)
        mfcc_std_mean = self.features.get('mfcc_std_mean', 0)
        
        ai_score = 0.0
        human_score = 0.0
        
        # Check MFCC variance
        if mfcc_var < self.THRESHOLDS['mfcc_variance_ai_max']:
            ai_score += 0.4
            self.ai_signals += 1
            self.indicators.append("Low spectral complexity detected")
        else:
            human_score += 0.4
            self.human_signals += 1
            self.indicators.append("Rich spectral complexity present")
        
        # Check MFCC delta
        if mfcc_delta_std < self.THRESHOLDS['mfcc_delta_std_ai_max']:
            ai_score += 0.3
            self.ai_signals += 1
            self.indicators.append("Limited dynamic speech patterns")
        else:
            human_score += 0.3
            self.human_signals += 1
        
        # Check MFCC std mean
        if mfcc_std_mean > 10:
            human_score += 0.3
            self.human_signals += 1
        else:
            ai_score += 0.2
        
        total = ai_score + human_score
        if total > 0:
            self.scores['mfcc'] = ai_score / total
        else:
            self.scores['mfcc'] = 0.5
    
    def _analyze_spectral(self):
        """Analyze spectral features"""
        flatness = self.features.get('spectral_flatness_mean', 0)
        centroid_std = self.features.get('spectral_centroid_std', 0)
        contrast = self.features.get('spectral_contrast_mean', 0)
        
        ai_score = 0.0
        human_score = 0.0
        
        # Check spectral flatness
        if flatness < self.THRESHOLDS['spectral_flatness_ai_min']:
            human_score += 0.35
            self.human_signals += 1
            self.indicators.append("Natural tonal quality detected")
        else:
            ai_score += 0.25
        
        # Check spectral centroid variation
        if centroid_std > self.THRESHOLDS['spectral_centroid_std_ai_max']:
            human_score += 0.35
            self.human_signals += 1
            self.indicators.append("Natural frequency variations present")
        else:
            ai_score += 0.3
            self.ai_signals += 1
        
        # Check spectral contrast
        if contrast > 20:
            human_score += 0.3
            self.human_signals += 1
        else:
            ai_score += 0.2
        
        total = ai_score + human_score
        if total > 0:
            self.scores['spectral'] = ai_score / total
        else:
            self.scores['spectral'] = 0.5
    
    def _analyze_temporal(self):
        """Analyze temporal features"""
        silence_ratio = self.features.get('silence_ratio', 0)
        rms_cv = self.features.get('rms_cv', 0)
        zcr_std = self.features.get('zcr_std', 0)
        
        ai_score = 0.0
        human_score = 0.0
        
        # Check silence ratio (humans have natural pauses)
        if self.THRESHOLDS['silence_ratio_human_min'] <= silence_ratio <= self.THRESHOLDS['silence_ratio_human_max']:
            human_score += 0.5
            self.human_signals += 1
            self.indicators.append("Natural breathing pauses detected")
        elif silence_ratio < self.THRESHOLDS['silence_ratio_human_min']:
            ai_score += 0.4
            self.ai_signals += 1
            self.indicators.append("Lack of natural breathing pauses")
        else:
            ai_score += 0.3
            self.indicators.append("Unusual pause patterns detected")
        
        # Check RMS coefficient of variation
        if rms_cv > 0.4:
            human_score += 0.3
            self.human_signals += 1
            self.indicators.append("Natural energy variations in speech")
        else:
            ai_score += 0.25
        
        # Check zero crossing rate variation
        if zcr_std > 0.02:
            human_score += 0.2
        else:
            ai_score += 0.15
        
        total = ai_score + human_score
        if total > 0:
            self.scores['temporal'] = ai_score / total
        else:
            self.scores['temporal'] = 0.5
    
    def _analyze_voice_quality(self):
        """Analyze voice quality features"""
        jitter = self.features.get('jitter', 0)
        shimmer = self.features.get('shimmer', 0)
        harmonic_ratio = self.features.get('harmonic_ratio', 0)
        
        ai_score = 0.0
        human_score = 0.0
        
        # Check jitter (humans have natural micro-variations)
        if jitter > self.THRESHOLDS['jitter_ai_max']:
            human_score += 0.4
            self.human_signals += 1
            self.indicators.append("Natural voice micro-variations detected")
        else:
            ai_score += 0.4
            self.ai_signals += 1
            self.indicators.append("Missing natural voice micro-variations")
        
        # Check shimmer
        if shimmer > self.THRESHOLDS['shimmer_ai_max']:
            human_score += 0.3
            self.human_signals += 1
            self.indicators.append("Natural amplitude variations present")
        else:
            ai_score += 0.3
            self.ai_signals += 1
        
        # Check harmonic ratio (AI often too clean)
        if harmonic_ratio > self.THRESHOLDS['harmonic_ratio_ai_min']:
            ai_score += 0.25
            self.indicators.append("Unusually clean audio signal")
        else:
            human_score += 0.25
            self.human_signals += 1
        
        total = ai_score + human_score
        if total > 0:
            self.scores['voice_quality'] = ai_score / total
        else:
            self.scores['voice_quality'] = 0.5
    
    def _calculate_final_score(self) -> float:
        """Calculate weighted final AI probability score"""
        total_score = 0.0
        total_weight = 0.0
        
        for category, weight in self.WEIGHTS.items():
            if category in self.scores:
                total_score += self.scores[category] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.5
    
    def _calculate_confidence_boost(self) -> float:
        """Calculate confidence boost based on signal agreement"""
        total_signals = self.ai_signals + self.human_signals
        
        if total_signals == 0:
            return 0.0
        
        # If most signals agree, boost confidence
        if self.ai_signals > self.human_signals:
            agreement_ratio = self.ai_signals / total_signals
        else:
            agreement_ratio = self.human_signals / total_signals
        
        # Boost ranges from 0 to 0.25 based on agreement
        if agreement_ratio > 0.75:
            return 0.25
        elif agreement_ratio > 0.65:
            return 0.15
        elif agreement_ratio > 0.55:
            return 0.08
        else:
            return 0.0
    
    def _generate_explanation(self, classification: str) -> str:
        """Generate detailed explanation"""
        
        if classification == "AI_GENERATED":
            ai_indicators = [
                ind for ind in self.indicators 
                if any(word in ind.lower() for word in [
                    'unnatural', 'lack', 'missing', 'unusual', 
                    'limited', 'low', 'synthetic', 'clean'
                ])
            ]
            
            if len(ai_indicators) >= 2:
                return f"{ai_indicators[0]} and {ai_indicators[1].lower()}"
            elif ai_indicators:
                return ai_indicators[0]
            
            explanations = {
                'pitch': "Unnatural pitch consistency and robotic speech patterns detected",
                'mfcc': "Synthetic spectral patterns identified in voice analysis",
                'spectral': "Artificial frequency distribution detected",
                'temporal': "Mechanical timing patterns without natural rhythm",
                'voice_quality': "Missing natural voice micro-variations and tremors"
            }
            
            max_category = max(self.scores, key=self.scores.get)
            return explanations.get(max_category, "Synthetic speech patterns detected")
        
        else:  # HUMAN
            human_indicators = [
                ind for ind in self.indicators 
                if any(word in ind.lower() for word in [
                    'natural', 'present', 'human', 'rich', 'wide', 'breathing'
                ])
            ]
            
            if len(human_indicators) >= 2:
                return f"{human_indicators[0]} and {human_indicators[1].lower()}"
            elif human_indicators:
                return human_indicators[0]
            
            explanations = {
                'pitch': "Natural pitch variation consistent with human speech",
                'mfcc': "Complex spectral patterns typical of human voice",
                'spectral': "Natural frequency characteristics detected",
                'temporal': "Organic speech rhythm with natural pauses",
                'voice_quality': "Natural voice micro-variations and breathing detected"
            }
            
            min_category = min(self.scores, key=self.scores.get)
            return explanations.get(min_category, "Natural human speech characteristics identified")