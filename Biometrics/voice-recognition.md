# 🎤 Voice/Speaker Recognition

Comprehensive guide to voice biometrics, speaker recognition systems, and speech-based authentication technologies.

**Last Updated:** 2025-06-20

## Table of Contents
- [Introduction](#introduction)
- [Voice Biometric Fundamentals](#voice-biometric-fundamentals)
- [Feature Extraction](#feature-extraction)
- [Speaker Recognition Approaches](#speaker-recognition-approaches)
- [Deep Learning Models](#deep-learning-models)
- [Text-Dependent vs Text-Independent](#text-dependent-vs-text-independent)
- [Frameworks & Tools](#frameworks--tools)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Implementation Guide](#implementation-guide)
- [Anti-Spoofing & Security](#anti-spoofing--security)
- [Real-World Applications](#real-world-applications)
- [Resources](#resources)

## Introduction

Voice biometrics uses unique characteristics of an individual's voice for identification and verification:
- **Behavioral & Physical**: Combines vocal tract shape and learned speech patterns
- **Non-intrusive**: Natural interaction method
- **Remote capable**: Works over phone/internet
- **Multi-factor**: Can combine with speech recognition
- **Language independent**: Works across languages

### Applications
- Call center authentication
- Voice assistants (Alexa, Siri, Google)
- Banking & financial services
- Smart home access control
- Forensic speaker identification
- Healthcare patient verification

## Voice Biometric Fundamentals

### Acoustic Features
```python
VOICE_CHARACTERISTICS = {
    # Physical (Physiological)
    'vocal_tract_length': 'Determined by throat and mouth',
    'vocal_fold_size': 'Affects pitch range',
    'nasal_cavity': 'Influences resonance',
    
    # Behavioral
    'speaking_rate': 'Words per minute',
    'prosody': 'Rhythm and intonation',
    'accent': 'Regional pronunciation',
    'articulation': 'Clarity of speech'
}
```

### Speech Production Model
```python
import numpy as np
import scipy.signal as signal

class SpeechProductionModel:
    def __init__(self, fs=16000):
        self.fs = fs  # Sampling frequency
        
    def generate_vowel(self, f0, formants, duration):
        """Generate synthetic vowel sound"""
        t = np.arange(0, duration, 1/self.fs)
        
        # Glottal source (voiced excitation)
        glottal = self.glottal_pulse(f0, t)
        
        # Vocal tract filter (formants)
        vocal_tract = self.formant_filter(formants)
        
        # Generate speech
        speech = signal.lfilter(vocal_tract[0], vocal_tract[1], glottal)
        
        # Add natural variations
        speech = self.add_jitter_shimmer(speech, f0)
        
        return speech
    
    def glottal_pulse(self, f0, t):
        """Liljencrants-Fant glottal model"""
        # Opening phase
        te = 0.4  # Opening quotient
        tp = 0.16  # Speed quotient
        
        period = 1/f0
        glottal = np.zeros_like(t)
        
        for i in range(int(len(t) * f0)):
            t0 = i * period
            mask = (t >= t0) & (t < t0 + period)
            t_norm = (t[mask] - t0) / period
            
            # LF model equations
            opening = t_norm < te
            closing = ~opening
            
            glottal[mask] = np.where(
                opening,
                0.5 * (1 - np.cos(np.pi * t_norm / te)),
                np.exp(-500 * (t_norm - te))
            )
        
        return glottal
    
    def formant_filter(self, formants):
        """Create formant filter"""
        # Formant frequencies and bandwidths
        b = 1
        a = 1
        
        for freq, bw in formants:
            # Convert to digital filter
            r = np.exp(-np.pi * bw / self.fs)
            theta = 2 * np.pi * freq / self.fs
            
            # Second-order resonator
            b_res = [1, -2*r*np.cos(theta), r**2]
            a_res = [1, 0, 0]
            
            b = signal.convolve(b, b_res)
            a = signal.convolve(a, a_res)
        
        return b, a
```

## Feature Extraction

### MFCC (Mel-frequency Cepstral Coefficients)
```python
import librosa
import numpy as np
from scipy.fftpack import dct

class MFCCExtractor:
    def __init__(self, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract(self, audio, sr=16000):
        """Extract MFCC features with deltas"""
        # Standard MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Delta coefficients (velocity)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Delta-delta (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack all features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return features.T  # (time, features)
    
    def extract_with_context(self, audio, sr=16000, context_size=9):
        """Extract MFCC with temporal context"""
        mfcc = self.extract(audio, sr)
        
        # Add context frames
        n_frames = mfcc.shape[0]
        n_features = mfcc.shape[1]
        context_features = []
        
        half_context = context_size // 2
        
        for i in range(n_frames):
            # Get context window
            start = max(0, i - half_context)
            end = min(n_frames, i + half_context + 1)
            
            # Pad if necessary
            context = np.zeros((context_size, n_features))
            actual_context = mfcc[start:end]
            
            offset = half_context - (i - start)
            context[offset:offset + len(actual_context)] = actual_context
            
            context_features.append(context.flatten())
        
        return np.array(context_features)
```

### Advanced Features
```python
class AdvancedFeatureExtractor:
    def __init__(self):
        self.feature_extractors = {
            'prosodic': self.extract_prosodic_features,
            'spectral': self.extract_spectral_features,
            'voice_quality': self.extract_voice_quality_features,
            'formants': self.extract_formant_features
        }
    
    def extract_all_features(self, audio, sr=16000):
        """Extract comprehensive feature set"""
        features = {}
        
        for name, extractor in self.feature_extractors.items():
            features[name] = extractor(audio, sr)
        
        return features
    
    def extract_prosodic_features(self, audio, sr):
        """Extract prosodic features"""
        # Fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        # Remove unvoiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            features = {
                'f0_mean': np.mean(f0_voiced),
                'f0_std': np.std(f0_voiced),
                'f0_max': np.max(f0_voiced),
                'f0_min': np.min(f0_voiced),
                'f0_range': np.max(f0_voiced) - np.min(f0_voiced),
                'jitter': self.compute_jitter(f0_voiced)
            }
        else:
            features = {k: 0 for k in ['f0_mean', 'f0_std', 'f0_max', 
                                       'f0_min', 'f0_range', 'jitter']}
        
        # Energy features
        energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        features.update({
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'shimmer': self.compute_shimmer(energy)
        })
        
        # Speaking rate (simplified)
        features['speaking_rate'] = self.estimate_speaking_rate(audio, sr)
        
        return features
    
    def extract_spectral_features(self, audio, sr):
        """Extract spectral features"""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Spectral flux
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        features = {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_flux_mean': np.mean(onset_env),
            'spectral_flux_std': np.std(onset_env),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr)
        }
        
        return features
    
    def extract_voice_quality_features(self, audio, sr):
        """Extract voice quality measures"""
        # Harmonics-to-Noise Ratio (HNR)
        hnr = self.compute_hnr(audio, sr)
        
        # Cepstral Peak Prominence (CPP)
        cpp = self.compute_cpp(audio, sr)
        
        # Glottal features
        glottal_features = self.extract_glottal_features(audio, sr)
        
        features = {
            'hnr': hnr,
            'cpp': cpp,
            **glottal_features
        }
        
        return features
    
    def extract_formant_features(self, audio, sr):
        """Extract formant frequencies"""
        # Pre-emphasis
        pre_emphasized = librosa.effects.preemphasis(audio)
        
        # LPC analysis
        lpc_order = 2 + sr // 1000
        lpc_coeffs = librosa.lpc(pre_emphasized, order=lpc_order)
        
        # Find formants from LPC
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]
        
        angles = np.angle(roots)
        frequencies = angles * (sr / (2 * np.pi))
        
        # Sort and select formants
        frequencies = sorted(frequencies[frequencies > 90])[:5]
        
        # Pad if needed
        while len(frequencies) < 5:
            frequencies.append(0)
        
        features = {
            f'f{i+1}': freq for i, freq in enumerate(frequencies[:4])
        }
        
        # Formant bandwidths (simplified)
        features.update({
            f'bw{i+1}': 100 + 50*i for i in range(4)
        })
        
        return features
    
    def compute_jitter(self, f0_values):
        """Compute jitter (F0 variation)"""
        if len(f0_values) < 2:
            return 0
        
        differences = np.abs(np.diff(f0_values))
        mean_period = np.mean(1/f0_values[:-1])
        
        jitter = np.mean(differences) / mean_period
        return jitter * 100  # Percentage
    
    def compute_shimmer(self, amplitude_values):
        """Compute shimmer (amplitude variation)"""
        if len(amplitude_values) < 2:
            return 0
        
        differences = np.abs(np.diff(amplitude_values))
        mean_amplitude = np.mean(amplitude_values[:-1])
        
        shimmer = np.mean(differences) / mean_amplitude
        return shimmer * 100  # Percentage
    
    def compute_hnr(self, audio, sr):
        """Harmonics-to-Noise Ratio"""
        # Autocorrelation method
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after zero lag
        min_period = int(sr / 500)  # 500 Hz max
        max_period = int(sr / 50)   # 50 Hz min
        
        peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
        
        if autocorr[peak_idx] > 0:
            hnr = 10 * np.log10(autocorr[peak_idx] / (autocorr[0] - autocorr[peak_idx]))
        else:
            hnr = 0
        
        return hnr
```

## Speaker Recognition Approaches

### i-vector System
```python
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.linalg import eigh

class IVectorSystem:
    def __init__(self, ubm_components=512, tv_dim=400):
        self.ubm_components = ubm_components
        self.tv_dim = tv_dim
        self.ubm = None
        self.T = None  # Total variability matrix
    
    def train_ubm(self, features_list):
        """Train Universal Background Model"""
        # Concatenate all features
        all_features = np.vstack(features_list)
        
        # Train GMM-UBM
        self.ubm = GaussianMixture(
            n_components=self.ubm_components,
            covariance_type='diag',
            max_iter=100,
            n_init=3
        )
        
        self.ubm.fit(all_features)
        
        return self.ubm
    
    def train_total_variability(self, features_list, n_iter=10):
        """Train total variability matrix T"""
        n_features = features_list[0].shape[1]
        n_supervectors = len(features_list)
        
        # Initialize T randomly
        self.T = np.random.randn(
            self.ubm_components * n_features,
            self.tv_dim
        ) * 0.01
        
        # EM algorithm for T matrix
        for iteration in range(n_iter):
            print(f"TV iteration {iteration + 1}/{n_iter}")
            
            # E-step: Estimate i-vectors
            ivectors = []
            for features in features_list:
                ivector = self.extract_ivector(features)
                ivectors.append(ivector)
            
            # M-step: Update T
            self.update_T_matrix(features_list, ivectors)
        
        return self.T
    
    def extract_ivector(self, features):
        """Extract i-vector from features"""
        # Compute sufficient statistics
        stats = self.compute_stats(features)
        
        # Extract i-vector
        # w = (I + T'Σ^(-1)NT)^(-1) T'Σ^(-1)F
        
        n_components = self.ubm_components
        n_features = features.shape[1]
        
        # Simplified extraction (assuming diagonal covariance)
        I = np.eye(self.tv_dim)
        TtSigmaInv = np.zeros((self.tv_dim, n_components * n_features))
        
        for c in range(n_components):
            idx = c * n_features
            cov_inv = 1.0 / self.ubm.covariances_[c]
            TtSigmaInv[:, idx:idx+n_features] = self.T[idx:idx+n_features].T * cov_inv
        
        # Compute i-vector
        L = I + np.dot(TtSigmaInv, self.T)
        L_inv = np.linalg.inv(L)
        
        # First-order stats
        F = stats['first_order'].flatten()
        
        ivector = np.dot(L_inv, np.dot(TtSigmaInv, F))
        
        return ivector
    
    def compute_stats(self, features):
        """Compute sufficient statistics"""
        # Zero-order statistics (soft counts)
        posteriors = self.ubm.predict_proba(features)
        N = np.sum(posteriors, axis=0)
        
        # First-order statistics
        F = np.zeros((self.ubm_components, features.shape[1]))
        
        for t in range(features.shape[0]):
            for c in range(self.ubm_components):
                F[c] += posteriors[t, c] * features[t]
        
        # Center around UBM means
        for c in range(self.ubm_components):
            F[c] -= N[c] * self.ubm.means_[c]
        
        return {
            'zero_order': N,
            'first_order': F
        }
    
    def plda_scoring(self, ivector1, ivector2):
        """PLDA scoring for i-vector comparison"""
        # Simplified PLDA scoring
        # In practice, use a proper PLDA implementation
        
        # Length normalization
        ivector1 = ivector1 / np.linalg.norm(ivector1)
        ivector2 = ivector2 / np.linalg.norm(ivector2)
        
        # Cosine similarity
        score = np.dot(ivector1, ivector2)
        
        return score
```

### x-vector System (Deep Learning)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class XVectorNet(nn.Module):
    """x-vector extractor network"""
    def __init__(self, input_dim=40, num_speakers=None):
        super(XVectorNet, self).__init__()
        
        # Frame-level layers
        self.tdnn1 = TDNN(input_dim, 512, context=[-2, -1, 0, 1, 2])
        self.tdnn2 = TDNN(512, 512, context=[-2, 0, 2])
        self.tdnn3 = TDNN(512, 512, context=[-3, 0, 3])
        self.tdnn4 = TDNN(512, 512, context=[0])
        self.tdnn5 = TDNN(512, 1500, context=[0])
        
        # Statistics pooling
        self.stats_pooling = StatsPooling()
        
        # Segment-level layers
        self.segment1 = nn.Linear(3000, 512)  # mean + std
        self.segment2 = nn.Linear(512, 512)
        
        # Output layer (for training)
        if num_speakers:
            self.output = nn.Linear(512, num_speakers)
        else:
            self.output = None
    
    def forward(self, x, return_embedding=False):
        # Frame-level processing
        x = F.relu(self.tdnn1(x))
        x = F.relu(self.tdnn2(x))
        x = F.relu(self.tdnn3(x))
        x = F.relu(self.tdnn4(x))
        x = F.relu(self.tdnn5(x))
        
        # Statistics pooling
        x = self.stats_pooling(x)
        
        # Segment-level processing
        x = F.relu(self.segment1(x))
        embedding = self.segment2(x)
        
        if return_embedding:
            return embedding
        
        # Speaker classification (training only)
        if self.output:
            x = self.output(embedding)
            return x, embedding
        else:
            return embedding

class TDNN(nn.Module):
    """Time-Delay Neural Network layer"""
    def __init__(self, input_dim, output_dim, context):
        super(TDNN, self).__init__()
        self.context = context
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.kernel = nn.Linear(input_dim * len(context), output_dim)
    
    def forward(self, x):
        # x shape: (batch, time, features)
        batch_size, T, _ = x.shape
        
        # Collect context frames
        outputs = []
        
        for t in range(T):
            # Get context window
            context_frames = []
            for c in self.context:
                idx = t + c
                if 0 <= idx < T:
                    context_frames.append(x[:, idx, :])
                else:
                    # Pad with zeros
                    context_frames.append(torch.zeros_like(x[:, 0, :]))
            
            # Concatenate context
            context_input = torch.cat(context_frames, dim=1)
            
            # Apply linear transformation
            output = self.kernel(context_input)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class StatsPooling(nn.Module):
    """Statistics pooling layer"""
    def forward(self, x):
        # x shape: (batch, time, features)
        
        # Compute mean
        mean = torch.mean(x, dim=1)
        
        # Compute standard deviation
        std = torch.std(x, dim=1)
        
        # Concatenate statistics
        stats = torch.cat([mean, std], dim=1)
        
        return stats
```

### ECAPA-TDNN (State-of-the-art)
```python
class ECAPATDNN(nn.Module):
    """ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation"""
    def __init__(self, input_size=80, channels=1024, emb_size=192):
        super(ECAPATDNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        
        # SE-Res2Blocks
        self.layer1 = SERes2Block(channels, channels, res2_scale=8, se_channels=128)
        self.layer2 = SERes2Block(channels, channels, res2_scale=8, se_channels=128)
        self.layer3 = SERes2Block(channels, channels, res2_scale=8, se_channels=128)
        
        # Multi-layer aggregation
        self.mha = nn.MultiheadAttention(channels * 3, num_heads=8)
        
        # Statistics pooling
        self.asp = AttentiveStatisticsPooling(channels * 3, bottleneck_dim=128)
        
        # Final layers
        self.fc = nn.Linear(channels * 6, emb_size)
        self.bn2 = nn.BatchNorm1d(emb_size)
    
    def forward(self, x):
        # Initial convolution
        x = x.transpose(1, 2)  # (B, F, T) -> (B, T, F)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # SE-Res2Blocks with skip connections
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Multi-layer feature aggregation
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Channel attention
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x, _ = self.mha(x, x, x)
        x = x.transpose(1, 2)  # Back to (B, C, T)
        
        # Attentive statistics pooling
        x = self.asp(x)
        
        # Final embedding
        x = self.fc(x)
        x = self.bn2(x)
        
        return x

class SERes2Block(nn.Module):
    """SE-Res2Net block"""
    def __init__(self, channels, out_channels, res2_scale=4, se_channels=128):
        super(SERes2Block, self).__init__()
        
        self.res2_scale = res2_scale
        self.channels = channels
        
        # Res2Net modules
        self.conv1 = nn.ModuleList([
            nn.Conv1d(channels // res2_scale, channels // res2_scale, 
                     kernel_size=3, padding=1)
            for _ in range(res2_scale - 1)
        ])
        
        # SE module
        self.se = SEModule(channels, se_channels)
        
        # Output
        self.conv2 = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        # Split input
        xs = torch.chunk(x, self.res2_scale, dim=1)
        
        # Res2Net
        y = xs[0]
        ys = [y]
        
        for i, conv in enumerate(self.conv1):
            y = y + xs[i + 1]
            y = F.relu(conv(y))
            ys.append(y)
        
        y = torch.cat(ys, dim=1)
        
        # SE
        y = self.se(y)
        
        # Output
        y = self.conv2(y)
        y = self.bn(y)
        
        # Residual connection
        return F.relu(y + x)

class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling"""
    def __init__(self, channels, bottleneck_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck_dim, channels, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        # x shape: (B, C, T)
        
        # Compute attention weights
        w = self.attention(x)
        
        # Weighted statistics
        mu = torch.sum(x * w, dim=2)
        
        # Weighted standard deviation
        var = torch.sum((x - mu.unsqueeze(2))**2 * w, dim=2)
        std = torch.sqrt(var.clamp(min=1e-5))
        
        # Concatenate statistics
        return torch.cat([mu, std], dim=1)
```

## Text-Dependent vs Text-Independent

### Text-Dependent Recognition
```python
class TextDependentSystem:
    """Recognition using specific passphrase"""
    
    def __init__(self, passphrase="My voice is my password"):
        self.passphrase = passphrase
        self.dtw = DTW()
        self.model = None
    
    def enroll_speaker(self, audio_samples, speaker_id):
        """Enroll speaker with multiple utterances of passphrase"""
        # Extract features from each sample
        features_list = []
        
        for audio in audio_samples:
            # Verify passphrase content (using ASR)
            if not self.verify_passphrase_content(audio):
                continue
            
            # Extract features
            features = self.extract_features(audio)
            features_list.append(features)
        
        # Create speaker template
        template = self.create_template(features_list)
        
        # Store template
        self.store_template(speaker_id, template)
        
        return template
    
    def verify_speaker(self, audio, claimed_id):
        """Verify speaker with passphrase"""
        # Check passphrase content
        if not self.verify_passphrase_content(audio):
            return False, 0.0
        
        # Extract features
        features = self.extract_features(audio)
        
        # Load template
        template = self.load_template(claimed_id)
        
        # DTW alignment and scoring
        score = self.dtw_matching(features, template)
        
        # Apply threshold
        threshold = 0.7
        accepted = score > threshold
        
        return accepted, score
    
    def create_template(self, features_list):
        """Create template from multiple utterances"""
        # DTW-based averaging
        if len(features_list) == 1:
            return features_list[0]
        
        # Use first as reference
        reference = features_list[0]
        
        # Align all others to reference
        aligned_features = [reference]
        
        for features in features_list[1:]:
            aligned = self.dtw.align(features, reference)
            aligned_features.append(aligned)
        
        # Average aligned features
        template = np.mean(aligned_features, axis=0)
        
        return template
    
    def dtw_matching(self, features1, features2):
        """DTW-based matching"""
        distance, path = self.dtw.compute(features1, features2)
        
        # Normalize by path length
        normalized_distance = distance / len(path)
        
        # Convert to similarity score
        score = np.exp(-normalized_distance / 10)
        
        return score
    
    def verify_passphrase_content(self, audio):
        """Verify spoken content matches passphrase"""
        # Simple implementation - in practice use ASR
        # Return True for demo
        return True

class DTW:
    """Dynamic Time Warping implementation"""
    
    def compute(self, seq1, seq2, distance_func=None):
        """Compute DTW distance and path"""
        if distance_func is None:
            distance_func = lambda x, y: np.linalg.norm(x - y)
        
        n, m = len(seq1), len(seq2)
        
        # Initialize cost matrix
        cost = np.inf * np.ones((n + 1, m + 1))
        cost[0, 0] = 0
        
        # Fill cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = distance_func(seq1[i-1], seq2[j-1])
                
                cost[i, j] = d + min(
                    cost[i-1, j],    # insertion
                    cost[i, j-1],    # deletion
                    cost[i-1, j-1]   # match
                )
        
        # Backtrack to find path
        path = self.backtrack(cost)
        
        return cost[n, m], path
    
    def backtrack(self, cost):
        """Find optimal path through cost matrix"""
        i, j = cost.shape[0] - 1, cost.shape[1] - 1
        path = [(i-1, j-1)]
        
        while i > 1 and j > 1:
            candidates = [
                (i-1, j-1, cost[i-1, j-1]),
                (i-1, j, cost[i-1, j]),
                (i, j-1, cost[i, j-1])
            ]
            
            i_new, j_new, _ = min(candidates, key=lambda x: x[2])
            i, j = i_new, j_new
            
            if i > 0 and j > 0:
                path.append((i-1, j-1))
        
        return list(reversed(path))
```

### Text-Independent Recognition
```python
class TextIndependentSystem:
    """Recognition regardless of spoken content"""
    
    def __init__(self, model_type='xvector'):
        self.model_type = model_type
        self.feature_extractor = MFCCExtractor()
        
        if model_type == 'xvector':
            self.model = XVectorNet(input_dim=40)
        elif model_type == 'ecapa':
            self.model = ECAPATDNN(input_size=80)
        
        self.embeddings_db = {}
    
    def extract_embedding(self, audio):
        """Extract speaker embedding from audio"""
        # Extract acoustic features
        if self.model_type == 'xvector':
            features = self.feature_extractor.extract(audio)
        else:
            # Use mel-spectrogram for ECAPA-TDNN
            features = self.extract_melspec(audio)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            if self.model_type == 'xvector':
                embedding = self.model(features_tensor, return_embedding=True)
            else:
                embedding = self.model(features_tensor)
        
        return embedding.squeeze().numpy()
    
    def enroll_speaker(self, audio_samples, speaker_id):
        """Enroll speaker from multiple audio samples"""
        embeddings = []
        
        for audio in audio_samples:
            # Check audio quality
            if self.check_audio_quality(audio):
                embedding = self.extract_embedding(audio)
                embeddings.append(embedding)
        
        if len(embeddings) == 0:
            return False
        
        # Average embeddings
        speaker_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)
        
        # Store
        self.embeddings_db[speaker_id] = speaker_embedding
        
        return True
    
    def verify_speaker(self, audio, claimed_id, threshold=0.7):
        """Verify speaker identity"""
        if claimed_id not in self.embeddings_db:
            return False, 0.0
        
        # Extract embedding
        test_embedding = self.extract_embedding(audio)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Compare with enrolled embedding
        enrolled_embedding = self.embeddings_db[claimed_id]
        
        # Cosine similarity
        similarity = np.dot(test_embedding, enrolled_embedding)
        
        # Decision
        accepted = similarity > threshold
        
        return accepted, similarity
    
    def identify_speaker(self, audio, threshold=0.5):
        """Identify speaker from enrolled speakers"""
        # Extract embedding
        test_embedding = self.extract_embedding(audio)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Compare with all enrolled speakers
        scores = {}
        
        for speaker_id, enrolled_embedding in self.embeddings_db.items():
            similarity = np.dot(test_embedding, enrolled_embedding)
            scores[speaker_id] = similarity
        
        # Find best match
        if scores:
            best_speaker = max(scores, key=scores.get)
            best_score = scores[best_speaker]
            
            if best_score > threshold:
                return best_speaker, best_score
        
        return None, 0.0
    
    def extract_melspec(self, audio, sr=16000):
        """Extract mel-spectrogram features"""
        # Compute mel-spectrogram
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=512,
            hop_length=160,
            n_mels=80,
            fmin=20,
            fmax=7600
        )
        
        # Convert to log scale
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        return log_melspec.T
    
    def check_audio_quality(self, audio):
        """Basic audio quality check"""
        # Check duration
        if len(audio) < 16000:  # Less than 1 second
            return False
        
        # Check energy
        energy = np.sum(audio ** 2) / len(audio)
        if energy < 1e-4:  # Too quiet
            return False
        
        # Check clipping
        if np.max(np.abs(audio)) > 0.99:
            return False
        
        return True
```

## Deep Learning Models

### End-to-End Speaker Verification
```python
class SpeakerVerificationNet(nn.Module):
    """End-to-end neural speaker verification"""
    
    def __init__(self, encoder_type='ecapa', loss_type='aam_softmax'):
        super(SpeakerVerificationNet, self).__init__()
        
        # Encoder network
        if encoder_type == 'ecapa':
            self.encoder = ECAPATDNN()
            embedding_size = 192
        else:
            self.encoder = XVectorNet()
            embedding_size = 512
        
        # Loss function
        if loss_type == 'aam_softmax':
            self.loss_fn = ArcMarginProduct(
                in_features=embedding_size,
                out_features=1000,  # num speakers
                s=30.0,
                m=0.2
            )
        elif loss_type == 'ge2e':
            self.loss_fn = GE2ELoss()
    
    def forward(self, x, labels=None):
        # Extract embeddings
        embeddings = self.encoder(x)
        
        if labels is not None:
            # Training mode
            outputs = self.loss_fn(embeddings, labels)
            return outputs, embeddings
        else:
            # Inference mode
            return embeddings

class ArcMarginProduct(nn.Module):
    """Additive Angular Margin Loss (ArcFace)"""
    
    def __init__(self, in_features, out_features, s=30.0, m=0.2):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
    
    def forward(self, input, label):
        # Normalize input and weights
        input = F.normalize(input)
        weight = F.normalize(self.weight)
        
        # Compute cosine
        cosine = F.linear(input, weight)
        
        # Compute sine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Easy margin
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot labels
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class GE2ELoss(nn.Module):
    """Generalized End-to-End Loss"""
    
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
    
    def forward(self, embeddings):
        """
        embeddings: (N, M, D) - N speakers, M utterances, D dimensions
        """
        N, M, D = embeddings.shape
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=2)
        
        # Calculate centroids
        centroids = torch.mean(embeddings, dim=1)  # (N, D)
        
        # Calculate similarity matrix
        similarity = torch.zeros(N, M, N)
        
        for j in range(N):
            # Exclude current utterance from centroid
            for k in range(M):
                centroid_exc = (torch.sum(embeddings[j], dim=0) - embeddings[j, k]) / (M - 1)
                centroid_exc = F.normalize(centroid_exc.unsqueeze(0), p=2, dim=1)
                
                # Similarity with all centroids
                similarity[j, k] = torch.mm(embeddings[j, k:k+1], centroids.t()).squeeze()
                
                # Update with excluded centroid for same speaker
                similarity[j, k, j] = torch.mm(embeddings[j, k:k+1], centroid_exc.t()).squeeze()
        
        # Scaled similarities
        similarity = self.w * similarity + self.b
        
        # Loss calculation
        label = torch.arange(N).unsqueeze(1).expand(-1, M).contiguous().view(-1)
        loss = F.cross_entropy(similarity.view(N*M, N), label)
        
        return loss
```

### Data Augmentation for Robustness
```python
class VoiceAugmentation:
    """Data augmentation for speaker recognition"""
    
    def __init__(self):
        self.augmentations = {
            'noise': self.add_noise,
            'reverb': self.add_reverb,
            'speed': self.change_speed,
            'pitch': self.change_pitch,
            'codec': self.simulate_codec
        }
    
    def augment(self, audio, sr=16000, aug_types=None):
        """Apply random augmentations"""
        if aug_types is None:
            aug_types = list(self.augmentations.keys())
        
        # Randomly select augmentation
        aug_type = np.random.choice(aug_types)
        
        return self.augmentations[aug_type](audio, sr)
    
    def add_noise(self, audio, sr, snr_db=20):
        """Add background noise"""
        # Generate noise
        noise = np.random.randn(len(audio))
        
        # Calculate scaling factor
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        # Add scaled noise
        noisy = audio + scale * noise
        
        return noisy
    
    def add_reverb(self, audio, sr):
        """Add reverberation"""
        # Simple reverb using convolution with IR
        # Generate synthetic impulse response
        ir_length = int(0.5 * sr)  # 0.5 second
        ir = np.random.exponential(0.1, ir_length)
        ir = ir / np.max(np.abs(ir))
        
        # Convolve
        reverb_audio = signal.convolve(audio, ir, mode='same')
        
        # Mix dry and wet
        mix = 0.3
        output = (1 - mix) * audio + mix * reverb_audio
        
        return output
    
    def change_speed(self, audio, sr, speed_factor=None):
        """Change speaking speed"""
        if speed_factor is None:
            speed_factor = np.random.uniform(0.9, 1.1)
        
        # Resample
        augmented = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        return augmented
    
    def change_pitch(self, audio, sr, n_steps=None):
        """Change pitch"""
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)
        
        # Pitch shift
        augmented = librosa.effects.pitch_shift(
            y=audio,
            sr=sr,
            n_steps=n_steps
        )
        
        return augmented
    
    def simulate_codec(self, audio, sr, codec='gsm'):
        """Simulate telephone codec"""
        # Downsample
        if codec == 'gsm':
            target_sr = 8000
        elif codec == 'amr':
            target_sr = 12200
        else:
            target_sr = 16000
        
        # Resample
        downsampled = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=target_sr
        )
        
        # Add quantization noise
        bit_depth = 8 if codec == 'gsm' else 13
        max_val = 2 ** (bit_depth - 1)
        
        quantized = np.round(downsampled * max_val) / max_val
        
        # Resample back
        upsampled = librosa.resample(
            y=quantized,
            orig_sr=target_sr,
            target_sr=sr
        )
        
        # Ensure same length
        if len(upsampled) > len(audio):
            upsampled = upsampled[:len(audio)]
        else:
            upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)))
        
        return upsampled
```

## Frameworks & Tools

### Open Source Libraries

#### SpeechBrain
```python
# Installation: pip install speechbrain

from speechbrain.pretrained import SpeakerRecognition

# Load pretrained model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Verify speakers
score, prediction = verification.verify_files(
    "speaker1.wav", 
    "speaker2.wav"
)

print(f"Same speaker: {prediction} (score: {score})")

# Extract embeddings
embeddings = verification.encode_file("speaker.wav")
```

#### Resemblyzer
```python
# Installation: pip install resemblyzer

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

# Load encoder
encoder = VoiceEncoder()

# Process audio files
wav1 = preprocess_wav(Path("speaker1.wav"))
wav2 = preprocess_wav(Path("speaker2.wav"))

# Extract embeddings
embed1 = encoder.embed_utterance(wav1)
embed2 = encoder.embed_utterance(wav2)

# Compare speakers
similarity = np.dot(embed1, embed2)
print(f"Speaker similarity: {similarity:.3f}")
```

#### PyAnnote Audio
```python
# Installation: pip install pyannote.audio

from pyannote.audio import Pipeline

# Load pretrained pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Apply pipeline
diarization = pipeline("audio.wav")

# Print speaker segments
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")
```

### Commercial APIs

#### Amazon Connect Voice ID
```python
import boto3

client = boto3.client('voice-id')

# Create domain
response = client.create_domain(
    Name='my-voice-domain',
    Description='Voice authentication domain',
    ServerSideEncryptionConfiguration={
        'KmsKeyId': 'alias/aws/voiceid'
    }
)

# Enroll speaker
enrollment = client.start_speaker_enrollment(
    DomainId=domain_id,
    SpeakerId=speaker_id
)

# Verify speaker
verification = client.start_speaker_verification(
    DomainId=domain_id,
    SpeakerId=speaker_id,
    SessionNameOrArn=session_arn
)
```

#### Microsoft Azure Speaker Recognition
```python
import azure.cognitiveservices.speech as speechsdk

# Create config
speech_config = speechsdk.SpeechConfig(
    subscription="YOUR_KEY",
    region="YOUR_REGION"
)

# Create verification profile
client = speechsdk.speaker.SpeakerRecognitionClient(speech_config)

profile = client.create_verification_profile(
    speechsdk.speaker.VoiceProfileType.TextIndependentVerification
)

# Enroll speaker
result = client.enroll_profile(
    profile,
    audio_config
)

# Verify speaker
result = client.verify_profile(
    profile,
    audio_config
)
```

## Datasets & Benchmarks

### Popular Datasets

#### VoxCeleb
| Dataset | Speakers | Utterances | Hours | Environment |
|---------|----------|------------|-------|-------------|
| **[VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)** | 1,251 | 153,516 | 352 | YouTube |
| **[VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)** | 6,112 | 1,128,246 | 2,442 | YouTube |

#### Other Datasets
- **[LibriSpeech](https://www.openslr.org/12/)** - 1000 hours clean speech
- **[VCTK](https://datashare.ed.ac.uk/handle/10283/3443)** - 110 English speakers
- **[CommonVoice](https://commonvoice.mozilla.org/)** - Multilingual dataset
- **[SITW](https://www.sri.com/computer-vision/speaker-recognition/)** - Speakers in the Wild
- **[CN-Celeb](https://cnceleb.org/)** - Chinese speakers

### Evaluation Protocols

#### Metrics
```python
def compute_eer(scores, labels):
    """Compute Equal Error Rate"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
    eer = fpr[np.argmin(np.abs(fnr - fpr))]
    
    return eer, eer_threshold

def compute_dcf(scores, labels, p_target=0.01, c_miss=1, c_fa=1):
    """Compute Detection Cost Function"""
    # Implementation of NIST DCF
    # ... (detailed implementation)
    pass

def compute_cavg(scores, labels):
    """Compute average cost (Cavg)"""
    # Used in NIST SRE evaluations
    # ... (detailed implementation)
    pass
```

## Implementation Guide

### Complete Speaker Recognition System
```python
import os
import json
import numpy as np
from datetime import datetime
import soundfile as sf

class SpeakerRecognitionSystem:
    def __init__(self, config_path='config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.feature_extractor = AdvancedFeatureExtractor()
        self.augmentor = VoiceAugmentation()
        
        # Load model
        if self.config['model_type'] == 'ecapa':
            self.model = ECAPATDNN()
            self.load_pretrained_weights()
        else:
            self.model = XVectorNet()
        
        # Speaker database
        self.speaker_db = {}
        self.load_database()
    
    def enroll_speaker(self, audio_paths, speaker_id, metadata=None):
        """Enroll new speaker"""
        print(f"Enrolling speaker: {speaker_id}")
        
        embeddings = []
        
        for audio_path in audio_paths:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Preprocess
            audio = self.preprocess_audio(audio, sr)
            
            # Quality check
            quality_score = self.check_voice_quality(audio)
            
            if quality_score < self.config['quality_threshold']:
                print(f"Low quality audio: {audio_path}")
                continue
            
            # Extract embedding
            embedding = self.extract_speaker_embedding(audio)
            embeddings.append(embedding)
        
        if len(embeddings) < self.config['min_enrollment_samples']:
            return {
                'success': False,
                'error': 'Insufficient quality samples'
            }
        
        # Create speaker model
        speaker_model = self.create_speaker_model(embeddings)
        
        # Store in database
        self.speaker_db[speaker_id] = {
            'model': speaker_model,
            'enrollment_date': str(datetime.now()),
            'num_samples': len(embeddings),
            'metadata': metadata or {}
        }
        
        # Save database
        self.save_database()
        
        return {
            'success': True,
            'speaker_id': speaker_id,
            'num_samples': len(embeddings)
        }
    
    def verify_speaker(self, audio_path, claimed_id):
        """Verify speaker identity"""
        # Check if speaker exists
        if claimed_id not in self.speaker_db:
            return {
                'verified': False,
                'error': 'Speaker not enrolled'
            }
        
        # Load and process audio
        audio, sr = sf.read(audio_path)
        audio = self.preprocess_audio(audio, sr)
        
        # Quality check
        quality_score = self.check_voice_quality(audio)
        
        if quality_score < self.config['quality_threshold'] * 0.8:
            return {
                'verified': False,
                'error': 'Low quality audio',
                'quality_score': quality_score
            }
        
        # Extract embedding
        test_embedding = self.extract_speaker_embedding(audio)
        
        # Compare with enrolled model
        speaker_model = self.speaker_db[claimed_id]['model']
        score = self.compute_similarity(test_embedding, speaker_model)
        
        # Anti-spoofing check
        spoofing_score = self.detect_spoofing(audio)
        
        # Decision
        threshold = self.config['verification_threshold']
        verified = (score > threshold) and (spoofing_score < 0.5)
        
        return {
            'verified': verified,
            'score': float(score),
            'spoofing_score': float(spoofing_score),
            'quality_score': quality_score
        }
    
    def identify_speaker(self, audio_path):
        """Identify speaker from enrolled speakers"""
        # Load and process audio
        audio, sr = sf.read(audio_path)
        audio = self.preprocess_audio(audio, sr)
        
        # Quality check
        quality_score = self.check_voice_quality(audio)
        
        if quality_score < self.config['quality_threshold'] * 0.8:
            return {
                'identified': False,
                'error': 'Low quality audio',
                'quality_score': quality_score
            }
        
        # Extract embedding
        test_embedding = self.extract_speaker_embedding(audio)
        
        # Compare with all enrolled speakers
        scores = {}
        
        for speaker_id, speaker_data in self.speaker_db.items():
            speaker_model = speaker_data['model']
            score = self.compute_similarity(test_embedding, speaker_model)
            scores[speaker_id] = score
        
        # Find best match
        if scores:
            best_speaker = max(scores, key=scores.get)
            best_score = scores[best_speaker]
            
            # Check threshold
            if best_score > self.config['identification_threshold']:
                # Anti-spoofing check
                spoofing_score = self.detect_spoofing(audio)
                
                if spoofing_score < 0.5:
                    return {
                        'identified': True,
                        'speaker_id': best_speaker,
                        'score': float(best_score),
                        'spoofing_score': float(spoofing_score)
                    }
        
        return {
            'identified': False,
            'speaker_id': None,
            'score': 0.0
        }
    
    def preprocess_audio(self, audio, sr):
        """Preprocess audio signal"""
        # Resample to target rate
        if sr != self.config['sample_rate']:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.config['sample_rate']
            )
        
        # Remove silence
        audio = self.remove_silence(audio)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # Apply pre-emphasis
        audio = librosa.effects.preemphasis(audio)
        
        return audio
    
    def remove_silence(self, audio, top_db=30):
        """Remove silence from audio"""
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) > 0:
            audio_trimmed = []
            for start, end in intervals:
                audio_trimmed.append(audio[start:end])
            
            audio = np.concatenate(audio_trimmed)
        
        return audio
    
    def check_voice_quality(self, audio):
        """Assess voice quality"""
        # Extract quality features
        features = self.feature_extractor.extract_voice_quality_features(
            audio,
            self.config['sample_rate']
        )
        
        # Simple quality score (0-100)
        score = 0
        
        # Check SNR
        if features['hnr'] > 10:
            score += 30
        elif features['hnr'] > 5:
            score += 20
        
        # Check duration
        duration = len(audio) / self.config['sample_rate']
        if duration > 2:
            score += 30
        elif duration > 1:
            score += 20
        
        # Check energy variation
        energy = librosa.feature.rms(y=audio)[0]
        if np.std(energy) > 0.01:
            score += 20
        
        # Additional checks
        score += min(20, features['cpp'] * 2)
        
        return min(100, score)
    
    def extract_speaker_embedding(self, audio):
        """Extract speaker embedding from audio"""
        # Extract features based on model type
        if self.config['model_type'] == 'ecapa':
            features = self.extract_melspec_features(audio)
        else:
            features = self.feature_extractor.extract(
                audio,
                self.config['sample_rate']
            )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Extract embedding
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'extract_embedding'):
                embedding = self.model.extract_embedding(features_tensor)
            else:
                embedding = self.model(features_tensor)
        
        # Post-process
        embedding = embedding.squeeze().numpy()
        
        # Length normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        return embedding
    
    def create_speaker_model(self, embeddings):
        """Create speaker model from embeddings"""
        embeddings = np.array(embeddings)
        
        # Simple approach: mean and covariance
        model = {
            'mean': np.mean(embeddings, axis=0),
            'std': np.std(embeddings, axis=0),
            'embeddings': embeddings  # Keep for advanced scoring
        }
        
        return model
    
    def compute_similarity(self, test_embedding, speaker_model):
        """Compute similarity score"""
        # Multiple scoring methods
        scores = []
        
        # Cosine similarity with mean
        cos_sim = np.dot(test_embedding, speaker_model['mean'])
        scores.append(cos_sim)
        
        # Probabilistic scoring
        if 'std' in speaker_model:
            # Mahalanobis-like distance
            diff = test_embedding - speaker_model['mean']
            std = speaker_model['std'] + 1e-6
            
            prob_score = np.exp(-0.5 * np.sum((diff / std) ** 2))
            scores.append(prob_score)
        
        # Average similarity with enrolled embeddings
        if 'embeddings' in speaker_model:
            similarities = [
                np.dot(test_embedding, emb)
                for emb in speaker_model['embeddings']
            ]
            avg_sim = np.mean(similarities)
            scores.append(avg_sim)
        
        # Combine scores
        final_score = np.mean(scores)
        
        return final_score
    
    def detect_spoofing(self, audio):
        """Detect voice spoofing attempts"""
        # Extract spoofing-specific features
        features = []
        
        # Spectral features
        spectral_features = self.feature_extractor.extract_spectral_features(
            audio,
            self.config['sample_rate']
        )
        features.extend(list(spectral_features.values()))
        
        # Cepstral features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config['sample_rate'],
            n_mfcc=20
        )
        
        # High-frequency cepstral coefficients (sensitive to artifacts)
        high_freq_energy = np.mean(np.abs(mfcc[15:]))
        features.append(high_freq_energy)
        
        # Simple threshold-based detection
        # In practice, use a trained classifier
        
        # Check for synthetic artifacts
        if high_freq_energy > 0.5:
            return 0.8  # Likely spoofed
        
        # Check spectral flatness (synthetic voices are often flatter)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        if np.mean(spectral_flatness) > 0.8:
            return 0.7
        
        return 0.2  # Likely genuine
    
    def save_database(self):
        """Save speaker database"""
        # Convert numpy arrays to lists for JSON
        db_serializable = {}
        
        for speaker_id, data in self.speaker_db.items():
            db_serializable[speaker_id] = {
                'model': {
                    'mean': data['model']['mean'].tolist(),
                    'std': data['model']['std'].tolist()
                },
                'enrollment_date': data['enrollment_date'],
                'num_samples': data['num_samples'],
                'metadata': data['metadata']
            }
        
        # Save to file
        db_path = self.config['database_path']
        with open(db_path, 'w') as f:
            json.dump(db_serializable, f, indent=2)
    
    def load_database(self):
        """Load speaker database"""
        db_path = self.config['database_path']
        
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                db_loaded = json.load(f)
            
            # Convert lists back to numpy arrays
            for speaker_id, data in db_loaded.items():
                model = {
                    'mean': np.array(data['model']['mean']),
                    'std': np.array(data['model']['std'])
                }
                
                self.speaker_db[speaker_id] = {
                    'model': model,
                    'enrollment_date': data['enrollment_date'],
                    'num_samples': data['num_samples'],
                    'metadata': data['metadata']
                }
    
    def update_speaker(self, speaker_id, new_audio_paths):
        """Update enrolled speaker with new samples"""
        if speaker_id not in self.speaker_db:
            return {'success': False, 'error': 'Speaker not enrolled'}
        
        # Extract embeddings from new samples
        new_embeddings = []
        
        for audio_path in new_audio_paths:
            audio, sr = sf.read(audio_path)
            audio = self.preprocess_audio(audio, sr)
            
            if self.check_voice_quality(audio) >= self.config['quality_threshold']:
                embedding = self.extract_speaker_embedding(audio)
                new_embeddings.append(embedding)
        
        if len(new_embeddings) == 0:
            return {'success': False, 'error': 'No quality samples'}
        
        # Update model
        old_model = self.speaker_db[speaker_id]['model']
        
        # Combine old and new embeddings
        if 'embeddings' in old_model:
            all_embeddings = list(old_model['embeddings']) + new_embeddings
        else:
            # Approximate from mean/std
            all_embeddings = [old_model['mean']] + new_embeddings
        
        # Recreate model
        new_model = self.create_speaker_model(all_embeddings)
        
        # Update database
        self.speaker_db[speaker_id]['model'] = new_model
        self.speaker_db[speaker_id]['num_samples'] += len(new_embeddings)
        
        self.save_database()
        
        return {
            'success': True,
            'new_samples': len(new_embeddings),
            'total_samples': self.speaker_db[speaker_id]['num_samples']
        }

# Configuration file example (config.json)
"""
{
    "model_type": "ecapa",
    "sample_rate": 16000,
    "quality_threshold": 60,
    "verification_threshold": 0.7,
    "identification_threshold": 0.6,
    "min_enrollment_samples": 3,
    "database_path": "speaker_database.json",
    "pretrained_weights": "models/ecapa_tdnn_voxceleb.pth"
}
"""

# Usage example
if __name__ == "__main__":
    # Initialize system
    system = SpeakerRecognitionSystem('config.json')
    
    # Enroll speaker
    result = system.enroll_speaker(
        audio_paths=['speaker1_1.wav', 'speaker1_2.wav', 'speaker1_3.wav'],
        speaker_id='john_doe',
        metadata={'age': 30, 'gender': 'male'}
    )
    print(f"Enrollment result: {result}")
    
    # Verify speaker
    result = system.verify_speaker('test_audio.wav', 'john_doe')
    print(f"Verification result: {result}")
    
    # Identify speaker
    result = system.identify_speaker('unknown_speaker.wav')
    print(f"Identification result: {result}")
```

## Anti-Spoofing & Security

### Voice Anti-Spoofing
```python
class VoiceAntiSpoofing:
    """Detect replay, synthesis, and conversion attacks"""
    
    def __init__(self):
        self.methods = {
            'replay': self.detect_replay_attack,
            'synthesis': self.detect_synthesis,
            'conversion': self.detect_voice_conversion,
            'deepfake': self.detect_deepfake_voice
        }
        
        # Load anti-spoofing model
        self.spoofing_model = self.load_spoofing_model()
    
    def detect_spoofing(self, audio, sr=16000):
        """Comprehensive spoofing detection"""
        results = {}
        
        # Apply all detection methods
        for attack_type, detector in self.methods.items():
            results[attack_type] = detector(audio, sr)
        
        # Combine scores
        spoofing_score = self.combine_scores(results)
        
        # Classification
        is_spoofed = spoofing_score > 0.5
        
        return {
            'is_spoofed': is_spoofed,
            'spoofing_score': spoofing_score,
            'attack_scores': results
        }
    
    def detect_replay_attack(self, audio, sr):
        """Detect replay attacks"""
        # Channel characteristics
        features = []
        
        # 1. Frequency response analysis
        freq_response = self.analyze_frequency_response(audio, sr)
        features.extend(freq_response)
        
        # 2. Reverberation patterns
        reverb_features = self.analyze_reverberation(audio, sr)
        features.extend(reverb_features)
        
        # 3. Background noise consistency
        noise_features = self.analyze_background_noise(audio, sr)
        features.extend(noise_features)
        
        # Simple scoring (use trained model in practice)
        replay_score = self.score_replay_features(features)
        
        return replay_score
    
    def detect_synthesis(self, audio, sr):
        """Detect synthetic speech"""
        # Artifacts from TTS systems
        features = []
        
        # 1. F0 patterns
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        if np.sum(voiced_flag) > 0:
            f0_voiced = f0[voiced_flag]
            
            # Check for unnatural stability
            f0_stability = np.std(np.diff(f0_voiced))
            features.append(f0_stability)
            
            # Check for quantization
            f0_quantization = self.detect_f0_quantization(f0_voiced)
            features.append(f0_quantization)
        
        # 2. Spectral artifacts
        spectral_features = self.analyze_spectral_artifacts(audio, sr)
        features.extend(spectral_features)
        
        # 3. Phase discontinuities
        phase_features = self.analyze_phase_discontinuities(audio, sr)
        features.extend(phase_features)
        
        # Score
        synthesis_score = self.score_synthesis_features(features)
        
        return synthesis_score
    
    def detect_voice_conversion(self, audio, sr):
        """Detect voice conversion attacks"""
        # Conversion artifacts
        features = []
        
        # 1. Formant analysis
        formant_features = self.analyze_formant_consistency(audio, sr)
        features.extend(formant_features)
        
        # 2. Spectral envelope smoothness
        envelope_features = self.analyze_spectral_envelope(audio, sr)
        features.extend(envelope_features)
        
        # 3. Residual analysis
        residual_features = self.analyze_glottal_residual(audio, sr)
        features.extend(residual_features)
        
        # Score
        conversion_score = self.score_conversion_features(features)
        
        return conversion_score
    
    def detect_deepfake_voice(self, audio, sr):
        """Detect deepfake voice using deep learning"""
        # Extract features for deepfake detection
        features = self.extract_deepfake_features(audio, sr)
        
        # Use pre-trained model
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            output = self.spoofing_model(features_tensor)
            deepfake_score = torch.sigmoid(output).item()
        
        return deepfake_score
    
    def analyze_frequency_response(self, audio, sr):
        """Analyze frequency response for replay detection"""
        # Compute spectrum
        D = librosa.stft(audio)
        mag = np.abs(D)
        
        # Average magnitude spectrum
        avg_spectrum = np.mean(mag, axis=1)
        
        # Look for:
        # 1. Band-limiting (replay through limited bandwidth)
        # 2. Comb filtering (multiple reflections)
        # 3. High-frequency roll-off
        
        features = []
        
        # High-frequency energy ratio
        freq_bins = librosa.fft_frequencies(sr=sr)
        hf_mask = freq_bins > 4000
        hf_energy = np.sum(avg_spectrum[hf_mask])
        total_energy = np.sum(avg_spectrum)
        hf_ratio = hf_energy / (total_energy + 1e-6)
        features.append(hf_ratio)
        
        # Spectral flatness in high frequencies
        hf_flatness = np.std(avg_spectrum[hf_mask]) / (np.mean(avg_spectrum[hf_mask]) + 1e-6)
        features.append(hf_flatness)
        
        return features
    
    def load_spoofing_model(self):
        """Load pre-trained anti-spoofing model"""
        # Simple CNN for demonstration
        model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
        # Load weights if available
        # model.load_state_dict(torch.load('antispoofing_model.pth'))
        
        model.eval()
        return model
```

### Liveness Detection
```python
class VoiceLivenessDetection:
    """Ensure live speaker presence"""
    
    def __init__(self):
        self.challenges = [
            'Please say the following numbers: {}',
            'Repeat this phrase: {}',
            'Count from {} to {}'
        ]
    
    def generate_challenge(self):
        """Generate random challenge"""
        import random
        
        challenge_type = random.choice(range(len(self.challenges)))
        
        if challenge_type == 0:
            # Random numbers
            numbers = [random.randint(0, 9) for _ in range(5)]
            challenge_text = ' '.join(map(str, numbers))
            prompt = self.challenges[0].format(challenge_text)
            expected = challenge_text
            
        elif challenge_type == 1:
            # Random phrase
            phrases = [
                "The quick brown fox",
                "Hello world today",
                "Voice authentication system"
            ]
            phrase = random.choice(phrases)
            prompt = self.challenges[1].format(phrase)
            expected = phrase
            
        else:
            # Counting
            start = random.randint(1, 5)
            end = start + random.randint(3, 5)
            prompt = self.challenges[2].format(start, end)
            expected = ' '.join(map(str, range(start, end + 1)))
        
        return {
            'prompt': prompt,
            'expected': expected,
            'challenge_id': self.generate_challenge_id()
        }
    
    def verify_challenge_response(self, audio, challenge):
        """Verify challenge response"""
        # 1. Verify content using ASR
        transcription = self.transcribe_audio(audio)
        content_match = self.verify_content(
            transcription,
            challenge['expected']
        )
        
        # 2. Verify timing
        response_time = len(audio) / 16000  # Assuming 16kHz
        timing_valid = self.verify_timing(
            response_time,
            challenge['expected']
        )
        
        # 3. Verify audio characteristics
        audio_valid = self.verify_audio_characteristics(audio)
        
        # Combined decision
        liveness_score = (
            0.5 * content_match +
            0.3 * timing_valid +
            0.2 * audio_valid
        )
        
        return {
            'liveness_score': liveness_score,
            'is_live': liveness_score > 0.7,
            'details': {
                'content_match': content_match,
                'timing_valid': timing_valid,
                'audio_valid': audio_valid
            }
        }
    
    def transcribe_audio(self, audio):
        """Simple ASR for challenge verification"""
        # In practice, use a proper ASR system
        # This is a placeholder
        return "1 2 3 4 5"
    
    def verify_content(self, transcription, expected):
        """Verify transcribed content matches expected"""
        # Normalize
        trans_words = transcription.lower().split()
        expected_words = expected.lower().split()
        
        # Calculate word error rate
        correct = sum(1 for t, e in zip(trans_words, expected_words) if t == e)
        
        if len(expected_words) > 0:
            accuracy = correct / len(expected_words)
        else:
            accuracy = 0
        
        return accuracy
    
    def verify_timing(self, response_time, expected_text):
        """Verify response timing is reasonable"""
        # Estimate expected duration
        num_words = len(expected_text.split())
        
        # Average speaking rate: 150 words/minute
        expected_duration = (num_words / 150) * 60
        
        # Allow 50% variation
        min_duration = expected_duration * 0.5
        max_duration = expected_duration * 1.5
        
        if min_duration <= response_time <= max_duration:
            return 1.0
        else:
            return 0.5
    
    def verify_audio_characteristics(self, audio):
        """Verify audio has live speech characteristics"""
        # Check for:
        # 1. Natural pauses
        # 2. Breathing sounds
        # 3. Micro-variations
        
        # Simplified check
        energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        
        # Check for natural energy variations
        energy_std = np.std(energy)
        
        if energy_std > 0.01:
            return 1.0
        else:
            return 0.5
    
    def generate_challenge_id(self):
        """Generate unique challenge ID"""
        import uuid
        return str(uuid.uuid4())
```

## Real-World Applications

### Call Center Integration
```python
class CallCenterVoiceAuth:
    """Voice authentication for call centers"""
    
    def __init__(self, recognition_system):
        self.recognition_system = recognition_system
        self.active_calls = {}
    
    def handle_incoming_call(self, call_id, phone_number):
        """Handle new incoming call"""
        # Initialize call session
        self.active_calls[call_id] = {
            'phone_number': phone_number,
            'start_time': datetime.now(),
            'audio_buffer': [],
            'auth_status': 'pending'
        }
        
        return self.get_greeting_message()
    
    def process_audio_chunk(self, call_id, audio_chunk):
        """Process real-time audio"""
        if call_id not in self.active_calls:
            return
        
        # Add to buffer
        self.active_calls[call_id]['audio_buffer'].append(audio_chunk)
        
        # Check if enough audio for authentication
        total_audio = np.concatenate(
            self.active_calls[call_id]['audio_buffer']
        )
        
        duration = len(total_audio) / 16000
        
        if duration > 3 and self.active_calls[call_id]['auth_status'] == 'pending':
            # Attempt authentication
            result = self.authenticate_caller(call_id, total_audio)
            
            if result['authenticated']:
                self.active_calls[call_id]['auth_status'] = 'authenticated'
                self.active_calls[call_id]['customer_id'] = result['customer_id']
                
                return {
                    'action': 'transfer_to_agent',
                    'message': f"Welcome back, {result['customer_name']}",
                    'customer_data': result['customer_data']
                }
            
            elif duration > 10:
                # Fallback to traditional authentication
                self.active_calls[call_id]['auth_status'] = 'failed'
                
                return {
                    'action': 'request_credentials',
                    'message': "Please provide your account number"
                }
        
        return None
    
    def authenticate_caller(self, call_id, audio):
        """Authenticate caller by voice"""
        # Get phone number
        phone_number = self.active_calls[call_id]['phone_number']
        
        # Look up possible customers by phone
        possible_customers = self.lookup_customers_by_phone(phone_number)
        
        if not possible_customers:
            return {'authenticated': False}
        
        # Try to identify speaker
        result = self.recognition_system.identify_speaker_from_list(
            audio,
            possible_customers
        )
        
        if result['identified']:
            customer_data = self.get_customer_data(result['speaker_id'])
            
            return {
                'authenticated': True,
                'customer_id': result['speaker_id'],
                'customer_name': customer_data['name'],
                'confidence': result['score'],
                'customer_data': customer_data
            }
        
        return {'authenticated': False}
```

### Smart Home Integration
```python
class SmartHomeVoiceControl:
    """Voice-controlled smart home with speaker identification"""
    
    def __init__(self, recognition_system, home_controller):
        self.recognition_system = recognition_system
        self.home_controller = home_controller
        self.user_preferences = {}
    
    def process_voice_command(self, audio):
        """Process voice command with user identification"""
        # Identify speaker
        speaker_result = self.recognition_system.identify_speaker(audio)
        
        if speaker_result['identified']:
            user_id = speaker_result['speaker_id']
            
            # Transcribe command
            command = self.transcribe_command(audio)
            
            # Execute with user context
            result = self.execute_command(command, user_id)
            
            return result
        else:
            return {
                'success': False,
                'message': 'Unknown user. Please enroll your voice first.'
            }
    
    def execute_command(self, command, user_id):
        """Execute command with user preferences"""
        # Parse command
        action = self.parse_command(command)
        
        # Get user preferences
        preferences = self.user_preferences.get(user_id, {})
        
        # Apply personalized settings
        if action['type'] == 'lights':
            if action['action'] == 'on':
                # Use user's preferred brightness
                brightness = preferences.get('light_brightness', 80)
                color = preferences.get('light_color', 'warm')
                
                self.home_controller.set_lights(
                    on=True,
                    brightness=brightness,
                    color=color
                )
                
                return {
                    'success': True,
                    'message': f'Lights on at {brightness}% brightness'
                }
        
        elif action['type'] == 'temperature':
            # Use user's preferred temperature
            preferred_temp = preferences.get(
                f'temp_{action["time"]}',
                22  # Default
            )
            
            self.home_controller.set_temperature(preferred_temp)
            
            return {
                'success': True,
                'message': f'Temperature set to {preferred_temp}°C'
            }
        
        return {'success': False, 'message': 'Command not recognized'}
```

## Resources

### Research Papers
- **[X-vectors: Robust DNN Embeddings](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)** - Snyder et al., 2018
- **[ECAPA-TDNN: Emphasized Channel Attention](https://arxiv.org/abs/2005.07143)** - Desplanques et al., 2020
- **[VoxCeleb: Large-scale Speaker Identification](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf)** - Nagrani et al., 2017
- **[End-to-End Text-Independent Speaker Verification](https://arxiv.org/abs/1806.11265)** - Chung et al., 2018

### Books & Tutorials
- "Fundamentals of Speaker Recognition" - Beigi
- "Speaker Recognition: A Tutorial" - Campbell, 1997
- [Kaldi Speaker Recognition Tutorial](https://kaldi-asr.org/doc/dnn.html)
- [SpeechBrain Tutorials](https://speechbrain.github.io/tutorial_speaker.html)

### Competitions & Challenges
- **[VoxCeleb Speaker Recognition Challenge](https://www.robots.ox.ac.uk/~vgg/data/voxsrc/)** - Annual
- **[NIST Speaker Recognition Evaluation](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation)** - SRE
- **[ASVspoof Challenge](https://www.asvspoof.org/)** - Anti-spoofing

### Open Source Projects
- **[Kaldi](https://github.com/kaldi-asr/kaldi)** - Speech recognition toolkit
- **[SpeechBrain](https://github.com/speechbrain/speechbrain)** - PyTorch speech toolkit
- **[WeNet](https://github.com/wenet-e2e/wespeaker)** - WeSpeaker toolkit
- **[PyAnnote](https://github.com/pyannote/pyannote-audio)** - Speaker diarization