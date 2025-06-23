# 👆 Fingerprint Recognition

Complete guide to fingerprint recognition technology, algorithms, sensors, and implementation strategies.

**Last Updated:** 2025-06-20

## Table of Contents
- [Introduction](#introduction)
- [Fingerprint Fundamentals](#fingerprint-fundamentals)
- [Sensor Technologies](#sensor-technologies)
- [Feature Extraction](#feature-extraction)
- [Matching Algorithms](#matching-algorithms)
- [Deep Learning Approaches](#deep-learning-approaches)
- [Quality Assessment](#quality-assessment)
- [Frameworks & SDKs](#frameworks--sdks)
- [Datasets & Standards](#datasets--standards)
- [Implementation Guide](#implementation-guide)
- [Mobile Integration](#mobile-integration)
- [Security & Anti-Spoofing](#security--anti-spoofing)
- [Resources](#resources)

## Introduction

Fingerprint recognition is the most mature and widely deployed biometric technology with applications in:
- Law enforcement (AFIS)
- Mobile device authentication
- Border control
- Access control systems
- Financial services
- National ID programs

### Advantages
- **Uniqueness**: No two fingerprints are identical
- **Permanence**: Patterns remain unchanged throughout life
- **Universality**: Everyone has fingerprints (except rare conditions)
- **Collectability**: Easy to capture
- **Performance**: High accuracy rates
- **Acceptability**: Widely accepted by users

## Fingerprint Fundamentals

### Ridge Patterns

#### Global Features (Classes)
1. **Arch** (5%)
   - Plain arch
   - Tented arch
2. **Loop** (65%)
   - Left loop
   - Right loop
3. **Whorl** (30%)
   - Plain whorl
   - Central pocket whorl
   - Double loop whorl
   - Accidental whorl

### Minutiae Types
```python
# Common minutiae types
MINUTIAE_TYPES = {
    'ENDING': 1,      # Ridge ending
    'BIFURCATION': 2, # Ridge bifurcation
    'ISLAND': 3,      # Short ridge
    'DOT': 4,         # Very short ridge
    'LAKE': 5,        # Enclosure
    'SPUR': 6,        # Hook
    'BRIDGE': 7,      # Ridge connection
    'DELTA': 8,       # Triangle-like region
    'CORE': 9         # Center of fingerprint
}
```

### Level of Features
1. **Level 1**: Pattern type (arch, loop, whorl)
2. **Level 2**: Minutiae points (endings, bifurcations)
3. **Level 3**: Pores, ridge contours, incipient ridges

## Sensor Technologies

### Optical Sensors
**[FTIR (Frustrated Total Internal Reflection)](https://en.wikipedia.org/wiki/Frustrated_total_internal_reflection)** - Traditional method
- High image quality
- Large capture area
- Resistant to ESD
- Bulky size

```python
# Optical sensor image enhancement
import cv2
import numpy as np

def enhance_optical_image(img):
    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return binary
```

### Capacitive Sensors
**[Capacitive Sensing](https://www.synaptics.com/)** - Most common in mobile
- Silicon chip technology
- Small form factor
- Good image quality
- Susceptible to moisture

### Ultrasonic Sensors
**[Qualcomm 3D Sonic](https://www.qualcomm.com/products/features/fingerprint-sensors)** - Latest technology
- Works through glass/metal
- 3D fingerprint data
- High security
- Expensive

### Thermal Sensors
**[Thermal Imaging](https://www.next-biometrics.com/)** - Temperature differential
- Works in harsh conditions
- No latent prints
- Power efficient
- Lower resolution

## Feature Extraction

### Minutiae Extraction
```python
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_bool

class MinutiaeExtractor:
    def __init__(self):
        self.minutiae_list = []
    
    def extract_minutiae(self, binary_image):
        # Skeletonize the image
        skeleton = skeletonize(img_as_bool(binary_image))
        
        # Find minutiae points
        minutiae = []
        rows, cols = skeleton.shape
        
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if skeleton[r, c] == 1:
                    # Get 3x3 neighborhood
                    neighbors = skeleton[r-1:r+2, c-1:c+2]
                    cn = self._compute_crossing_number(neighbors)
                    
                    if cn == 1:  # Ridge ending
                        minutiae.append({
                            'type': 'ending',
                            'x': c,
                            'y': r,
                            'angle': self._compute_angle(skeleton, r, c)
                        })
                    elif cn == 3:  # Bifurcation
                        minutiae.append({
                            'type': 'bifurcation',
                            'x': c,
                            'y': r,
                            'angle': self._compute_angle(skeleton, r, c)
                        })
        
        return minutiae
    
    def _compute_crossing_number(self, neighbors):
        # Crossing number computation
        p = neighbors.flatten()
        return 0.5 * sum(abs(p[i] - p[(i+1)%8]) for i in range(8))
    
    def _compute_angle(self, skeleton, r, c):
        # Compute ridge orientation at minutiae point
        window_size = 16
        r_start = max(0, r - window_size//2)
        r_end = min(skeleton.shape[0], r + window_size//2)
        c_start = max(0, c - window_size//2)
        c_end = min(skeleton.shape[1], c + window_size//2)
        
        window = skeleton[r_start:r_end, c_start:c_end]
        
        # Gradient computation
        gy, gx = np.gradient(window.astype(float))
        angle = np.arctan2(2*np.sum(gx*gy), np.sum(gx**2 - gy**2)) / 2
        
        return angle
```

### Ridge Orientation & Frequency
```python
def compute_orientation_field(image, block_size=16):
    """Compute local ridge orientation"""
    height, width = image.shape
    orientations = np.zeros((height//block_size, width//block_size))
    
    # Sobel gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    for i in range(0, height-block_size, block_size):
        for j in range(0, width-block_size, block_size):
            # Extract block
            gx_block = gx[i:i+block_size, j:j+block_size]
            gy_block = gy[i:i+block_size, j:j+block_size]
            
            # Compute orientation
            gxx = np.sum(gx_block * gx_block)
            gyy = np.sum(gy_block * gy_block)
            gxy = np.sum(gx_block * gy_block)
            
            angle = 0.5 * np.arctan2(2*gxy, gxx-gyy)
            orientations[i//block_size, j//block_size] = angle
    
    return orientations

def compute_ridge_frequency(image, orientations, block_size=32):
    """Compute local ridge frequency"""
    frequencies = np.zeros_like(orientations)
    
    for i in range(orientations.shape[0]):
        for j in range(orientations.shape[1]):
            # Extract oriented window
            angle = orientations[i, j]
            block = image[i*block_size:(i+1)*block_size, 
                         j*block_size:(j+1)*block_size]
            
            # Rotate block to horizontal
            rotated = rotate_block(block, -angle)
            
            # Project to x-axis
            projection = np.sum(rotated, axis=0)
            
            # Find peaks (ridges)
            peaks = find_peaks(projection)
            
            if len(peaks) > 1:
                # Average distance between peaks
                avg_distance = np.mean(np.diff(peaks))
                frequencies[i, j] = 1.0 / avg_distance
    
    return frequencies
```

## Matching Algorithms

### Minutiae-based Matching
```python
import numpy as np
from scipy.spatial.distance import cdist

class MinutiaeMatcher:
    def __init__(self, threshold_distance=15, threshold_angle=np.pi/6):
        self.threshold_distance = threshold_distance
        self.threshold_angle = threshold_angle
    
    def match(self, minutiae1, minutiae2):
        """Match two sets of minutiae"""
        if len(minutiae1) == 0 or len(minutiae2) == 0:
            return 0.0
        
        # Extract coordinates
        coords1 = np.array([(m['x'], m['y']) for m in minutiae1])
        coords2 = np.array([(m['x'], m['y']) for m in minutiae2])
        
        # Try different alignments
        best_score = 0
        for ref_idx in range(min(5, len(minutiae1))):
            # Use reference minutia for alignment
            ref = minutiae1[ref_idx]
            
            for target_idx in range(len(minutiae2)):
                target = minutiae2[target_idx]
                
                # Only match same type
                if ref['type'] != target['type']:
                    continue
                
                # Compute transformation
                dx = target['x'] - ref['x']
                dy = target['y'] - ref['y']
                dtheta = target['angle'] - ref['angle']
                
                # Transform minutiae1 to align with minutiae2
                transformed = self._transform_minutiae(minutiae1, dx, dy, dtheta)
                
                # Count matches
                matches = self._count_matches(transformed, minutiae2)
                score = matches / max(len(minutiae1), len(minutiae2))
                
                if score > best_score:
                    best_score = score
        
        return best_score
    
    def _transform_minutiae(self, minutiae, dx, dy, dtheta):
        """Apply transformation to minutiae set"""
        transformed = []
        cos_theta = np.cos(dtheta)
        sin_theta = np.sin(dtheta)
        
        for m in minutiae:
            # Rotate and translate
            x_new = cos_theta * m['x'] - sin_theta * m['y'] + dx
            y_new = sin_theta * m['x'] + cos_theta * m['y'] + dy
            angle_new = m['angle'] + dtheta
            
            transformed.append({
                'type': m['type'],
                'x': x_new,
                'y': y_new,
                'angle': angle_new
            })
        
        return transformed
    
    def _count_matches(self, minutiae1, minutiae2):
        """Count matching minutiae pairs"""
        matches = 0
        
        for m1 in minutiae1:
            for m2 in minutiae2:
                if m1['type'] != m2['type']:
                    continue
                
                # Check spatial distance
                distance = np.sqrt((m1['x']-m2['x'])**2 + (m1['y']-m2['y'])**2)
                if distance > self.threshold_distance:
                    continue
                
                # Check angle difference
                angle_diff = abs(m1['angle'] - m2['angle'])
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                if angle_diff > self.threshold_angle:
                    continue
                
                matches += 1
                break
        
        return matches
```

### Pattern-based Matching
```python
from scipy.fftpack import fft2, ifft2, fftshift

def correlation_matching(template, query):
    """FFT-based correlation matching"""
    # Ensure same size
    h, w = max(template.shape[0], query.shape[0]), max(template.shape[1], query.shape[1])
    template_pad = np.pad(template, ((0, h-template.shape[0]), (0, w-template.shape[1])))
    query_pad = np.pad(query, ((0, h-query.shape[0]), (0, w-query.shape[1])))
    
    # FFT correlation
    f_template = fft2(template_pad)
    f_query = fft2(query_pad)
    correlation = ifft2(f_template * np.conj(f_query))
    correlation = fftshift(np.abs(correlation))
    
    # Find peak
    peak_value = np.max(correlation)
    
    # Normalize
    score = peak_value / (np.sqrt(np.sum(template**2) * np.sum(query**2)))
    
    return score
```

## Deep Learning Approaches

### CNN-based Feature Extraction
```python
import torch
import torch.nn as nn

class FingerprintCNN(nn.Module):
    def __init__(self, num_classes=None):
        super(FingerprintCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 128)  # 128-D embedding
        )
        
        # Classification layer (optional)
        self.classifier = nn.Linear(128, num_classes) if num_classes else None
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get embedding
        embedding = self.embedding(x)
        
        if self.classifier:
            # Classification
            output = self.classifier(embedding)
            return output, embedding
        else:
            return embedding

# Siamese Network for verification
class SiameseFingerprintNet(nn.Module):
    def __init__(self):
        super(SiameseFingerprintNet, self).__init__()
        self.backbone = FingerprintCNN()
        
    def forward(self, x1, x2):
        # Get embeddings
        embed1 = self.backbone(x1)
        embed2 = self.backbone(x2)
        
        # Compute distance
        distance = torch.nn.functional.pairwise_distance(embed1, embed2)
        
        return distance
```

### FingerNet Implementation
```python
# Based on "FingerNet: An Unified Deep Network for Fingerprint Minutiae Extraction"
class FingerNet(nn.Module):
    def __init__(self):
        super(FingerNet, self).__init__()
        
        # Shared convolutional layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Orientation estimation branch
        self.orientation_branch = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 90, 1)  # 90 orientation bins
        )
        
        # Segmentation branch
        self.segmentation_branch = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1)  # Background/foreground
        )
        
        # Enhancement branch
        self.enhancement_branch = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Enhanced image
        )
        
        # Minutiae extraction branch
        self.minutiae_branch = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, 1)  # No minutia, ending, bifurcation
        )
    
    def forward(self, x):
        # Shared features
        features = self.shared_conv(x)
        
        # Multi-task outputs
        orientation = self.orientation_branch(features)
        segmentation = self.segmentation_branch(features)
        enhancement = self.enhancement_branch(features)
        minutiae = self.minutiae_branch(features)
        
        return {
            'orientation': orientation,
            'segmentation': segmentation,
            'enhancement': enhancement,
            'minutiae': minutiae
        }
```

## Quality Assessment

### NFIQ 2.0 Implementation
```python
class FingerprintQualityAssessment:
    def __init__(self):
        self.quality_features = [
            'ridge_valley_uniformity',
            'ridge_valley_clarity',
            'orientation_certainty',
            'orientation_flow',
            'minutiae_count',
            'minutiae_quality'
        ]
    
    def assess_quality(self, fingerprint_image):
        """Compute NFIQ 2.0 quality score"""
        features = {}
        
        # Ridge-valley uniformity
        features['ridge_valley_uniformity'] = self._compute_rv_uniformity(fingerprint_image)
        
        # Ridge-valley clarity
        features['ridge_valley_clarity'] = self._compute_rv_clarity(fingerprint_image)
        
        # Orientation certainty
        orientations = compute_orientation_field(fingerprint_image)
        features['orientation_certainty'] = self._compute_orientation_certainty(orientations)
        
        # Orientation flow
        features['orientation_flow'] = self._compute_orientation_flow(orientations)
        
        # Minutiae features
        minutiae = MinutiaeExtractor().extract_minutiae(fingerprint_image)
        features['minutiae_count'] = len(minutiae)
        features['minutiae_quality'] = self._assess_minutiae_quality(minutiae, fingerprint_image)
        
        # Compute overall quality score (0-100)
        quality_score = self._compute_overall_score(features)
        
        return quality_score, features
    
    def _compute_rv_uniformity(self, image):
        """Compute ridge-valley uniformity metric"""
        # Local standard deviation
        kernel_size = 16
        local_std = cv2.blur((image - cv2.blur(image, (kernel_size, kernel_size)))**2, 
                            (kernel_size, kernel_size))**0.5
        
        # Uniformity score
        uniformity = 1.0 - (np.std(local_std) / (np.mean(local_std) + 1e-7))
        return uniformity
    
    def _compute_rv_clarity(self, image):
        """Compute ridge-valley clarity metric"""
        # Frequency domain analysis
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Ring filter for ridge frequencies (5-13 ridges per inch)
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        # Create ring mask
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Typical ridge frequencies
        inner_radius = 20
        outer_radius = 60
        ring_mask = (r >= inner_radius) & (r <= outer_radius)
        
        # Energy in ridge frequency band
        ridge_energy = np.sum(magnitude_spectrum[ring_mask]**2)
        total_energy = np.sum(magnitude_spectrum**2)
        
        clarity = ridge_energy / (total_energy + 1e-7)
        return clarity
    
    def _compute_orientation_certainty(self, orientations):
        """Compute orientation field certainty"""
        # Gradient of orientation field
        gy, gx = np.gradient(orientations)
        coherence = 1.0 - np.mean(np.sqrt(gx**2 + gy**2))
        return coherence
    
    def _compute_orientation_flow(self, orientations):
        """Compute smoothness of orientation field"""
        # Local orientation consistency
        flow_score = 0
        h, w = orientations.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = orientations[i, j]
                neighbors = orientations[i-1:i+2, j-1:j+2].flatten()
                
                # Angular differences
                diffs = np.abs(neighbors - center)
                diffs = np.minimum(diffs, 2*np.pi - diffs)
                
                flow_score += np.exp(-np.mean(diffs))
        
        return flow_score / ((h-2) * (w-2))
    
    def _assess_minutiae_quality(self, minutiae, image):
        """Assess quality of detected minutiae"""
        if len(minutiae) == 0:
            return 0.0
        
        quality_scores = []
        
        for m in minutiae:
            # Local image quality around minutia
            x, y = int(m['x']), int(m['y'])
            
            # Extract local patch
            patch_size = 32
            y_start = max(0, y - patch_size//2)
            y_end = min(image.shape[0], y + patch_size//2)
            x_start = max(0, x - patch_size//2)
            x_end = min(image.shape[1], x + patch_size//2)
            
            patch = image[y_start:y_end, x_start:x_end]
            
            # Local clarity
            local_clarity = np.std(patch) / (np.mean(patch) + 1e-7)
            quality_scores.append(local_clarity)
        
        return np.mean(quality_scores)
    
    def _compute_overall_score(self, features):
        """Compute overall quality score"""
        # Weights learned from NFIQ 2.0 training
        weights = {
            'ridge_valley_uniformity': 0.15,
            'ridge_valley_clarity': 0.20,
            'orientation_certainty': 0.15,
            'orientation_flow': 0.15,
            'minutiae_count': 0.20,
            'minutiae_quality': 0.15
        }
        
        # Normalize features
        normalized_features = {}
        for key, value in features.items():
            if key == 'minutiae_count':
                # Optimal minutiae count around 40-80
                normalized_features[key] = np.exp(-((value - 60)**2) / 1000)
            else:
                normalized_features[key] = np.clip(value, 0, 1)
        
        # Weighted sum
        score = sum(weights[k] * normalized_features[k] for k in weights)
        
        # Convert to 0-100 scale
        return int(score * 100)
```

## Frameworks & SDKs

### Open Source

#### NIST Biometric Image Software (NBIS)
```bash
# Install NBIS
wget https://www.nist.gov/file/387306
tar -xzf nbis-v5.0.0.tar.gz
cd nbis
./setup.sh /usr/local/nbis
make config
make it
```

Usage example:
```python
import subprocess

def extract_minutiae_nbis(image_path):
    """Extract minutiae using NBIS mindtct"""
    # Run mindtct
    cmd = f"mindtct {image_path} output"
    subprocess.run(cmd.split())
    
    # Read minutiae file
    minutiae = []
    with open("output.min", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                minutiae.append({
                    'x': int(parts[0]),
                    'y': int(parts[1]),
                    'angle': float(parts[2]),
                    'quality': int(parts[3]),
                    'type': parts[4]
                })
    
    return minutiae
```

#### SourceAFIS
**[SourceAFIS](https://sourceafis.machinezoo.com/)** - Java/.NET library
```python
# Python wrapper
from sourceafis import FingerprintTemplate, FingerprintMatcher

# Load fingerprint
with open("fingerprint.png", "rb") as f:
    template = FingerprintTemplate(f.read())

# Match fingerprints
matcher = FingerprintMatcher(template)
score = matcher.match(another_template)
```

#### OpenCV Fingerprint
```python
import cv2
import numpy as np

class OpenCVFingerprint:
    def __init__(self):
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=500)
        # FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=12,
                           key_size=20,
                           multi_probe_level=2)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def extract_features(self, image):
        """Extract ORB features"""
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_fingerprints(self, desc1, desc2):
        """Match two fingerprints using ORB features"""
        if desc1 is None or desc2 is None:
            return 0.0
        
        # Find matches
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Calculate match score
        score = len(good_matches) / max(len(desc1), len(desc2))
        return score
```

### Commercial SDKs

#### Neurotechnology VeriFinger
```python
# VeriFinger SDK Python example
from neurotec.biometrics import NFinger, NSubject
from neurotec.biometrics.client import NBiometricClient

class VeriFingerSDK:
    def __init__(self):
        self.client = NBiometricClient()
        self.client.fingerprinting_extraction_type = "minutiae"
    
    def extract_template(self, image_path):
        """Extract fingerprint template"""
        subject = NSubject()
        finger = NFinger()
        finger.image = NImage.from_file(image_path)
        subject.fingers.add(finger)
        
        # Extract template
        status = self.client.create_template(subject)
        
        if status == NBiometricStatus.OK:
            return subject.template
        else:
            raise Exception(f"Template extraction failed: {status}")
    
    def match_templates(self, template1, template2):
        """Match two templates"""
        subject1 = NSubject()
        subject1.template = template1
        
        subject2 = NSubject()
        subject2.template = template2
        
        # Verify
        status = self.client.verify(subject1, subject2)
        
        if status == NBiometricStatus.OK:
            return subject1.matching_results[0].score
        else:
            return 0
```

## Datasets & Standards

### Public Datasets

#### FVC (Fingerprint Verification Competition)
| Dataset | Sensors | Size | Resolution | Notes |
|---------|---------|------|------------|-------|
| **[FVC2000](https://biolab.csr.unibo.it/fvcongoing/UI/Form/Home.aspx)** | 4 types | 880 images | 300-500 dpi | Low quality |
| **[FVC2002](https://biolab.csr.unibo.it/fvcongoing/UI/Form/Home.aspx)** | 4 types | 880 images | 500 dpi | Better quality |
| **[FVC2004](https://biolab.csr.unibo.it/fvcongoing/UI/Form/Home.aspx)** | 4 types | 880 images | 500 dpi | Distortions |
| **[FVC2006](https://biolab.csr.unibo.it/fvcongoing/UI/Form/Home.aspx)** | 4 types | 1680 images | 250-569 dpi | Various quality |

#### NIST Datasets
- **[NIST SD4](https://www.nist.gov/srd/nist-special-database-4)** - 2000 8-bit gray scale images
- **[NIST SD14](https://www.nist.gov/srd/nist-special-database-14)** - 27,000 images
- **[NIST SD27](https://www.nist.gov/srd/nist-special-database-27)** - Latent fingerprints
- **[NIST SD302](https://www.nist.gov/itl/iad/image-group/nist-special-database-302)** - Nail-to-nail captures

### Standards

#### ISO/IEC Standards
- **ISO/IEC 19794-2**: Fingerprint minutiae data format
- **ISO/IEC 19794-4**: Fingerprint image data format
- **ISO/IEC 19794-8**: Fingerprint pattern skeletal data
- **ISO/IEC 29109**: Conformance testing for 19794-2

#### ANSI/NIST Standards
- **ANSI/NIST-ITL 1-2011**: Data format for information interchange
- **ANSI INCITS 378**: Fingerprint minutiae format
- **ANSI INCITS 381**: Fingerprint image quality

### Minutiae Format
```python
class MinutiaeISO19794:
    """ISO/IEC 19794-2 minutiae format"""
    
    def __init__(self):
        self.format_identifier = b'FMR\x00'
        self.version = b'020\x00'
        self.record_length = 0
        self.capture_device_id = 0
        self.image_size_x = 0
        self.image_size_y = 0
        self.x_resolution = 197  # 500 dpi in pixels/cm
        self.y_resolution = 197
        self.minutiae = []
    
    def add_minutia(self, x, y, angle, minutia_type, quality=60):
        """Add minutia point"""
        # Convert angle to ISO format (2 degree units)
        iso_angle = int(angle * 180 / np.pi / 2)
        
        minutia = {
            'x': x,
            'y': y,
            'angle': iso_angle,
            'type': minutia_type,
            'quality': quality
        }
        self.minutiae.append(minutia)
    
    def encode(self):
        """Encode to ISO format"""
        data = bytearray()
        
        # Header
        data.extend(self.format_identifier)
        data.extend(self.version)
        data.extend(struct.pack('>I', 0))  # Record length (update later)
        data.extend(struct.pack('>H', self.capture_device_id))
        data.extend(struct.pack('>H', self.image_size_x))
        data.extend(struct.pack('>H', self.image_size_y))
        data.extend(struct.pack('>H', self.x_resolution))
        data.extend(struct.pack('>H', self.y_resolution))
        data.append(len(self.minutiae))
        
        # Minutiae data
        for m in self.minutiae:
            # Type and X coordinate
            type_x = (m['type'] << 14) | (m['x'] & 0x3FFF)
            data.extend(struct.pack('>H', type_x))
            
            # Y coordinate
            data.extend(struct.pack('>H', m['y'] & 0x3FFF))
            
            # Angle and quality
            angle_quality = (m['angle'] << 6) | (m['quality'] & 0x3F)
            data.append(angle_quality)
        
        # Update record length
        record_length = len(data)
        data[8:12] = struct.pack('>I', record_length)
        
        return bytes(data)
```

## Implementation Guide

### Complete Fingerprint System
```python
import cv2
import numpy as np
from typing import Tuple, List, Dict
import pickle

class FingerprintRecognitionSystem:
    def __init__(self, use_deep_learning=False):
        self.use_deep_learning = use_deep_learning
        self.database = {}
        
        if use_deep_learning:
            # Load pre-trained CNN model
            self.model = FingerprintCNN()
            self.model.load_state_dict(torch.load('fingerprint_model.pth'))
            self.model.eval()
        else:
            # Traditional approach
            self.extractor = MinutiaeExtractor()
            self.matcher = MinutiaeMatcher()
            self.quality_assessor = FingerprintQualityAssessment()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess fingerprint image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Enhance image
        enhanced = self.enhance_fingerprint(image)
        
        return enhanced
    
    def enhance_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Enhance fingerprint image using Gabor filters"""
        # Compute orientation field
        orientations = compute_orientation_field(image, block_size=16)
        
        # Compute frequency field
        frequencies = compute_ridge_frequency(image, orientations)
        
        # Apply Gabor filter bank
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        block_size = 16
        for i in range(0, image.shape[0] - block_size, block_size):
            for j in range(0, image.shape[1] - block_size, block_size):
                # Get local orientation and frequency
                oi = i // block_size
                oj = j // block_size
                
                if oi < orientations.shape[0] and oj < orientations.shape[1]:
                    orientation = orientations[oi, oj]
                    frequency = frequencies[oi, oj]
                    
                    # Create Gabor kernel
                    kernel = cv2.getGaborKernel(
                        (block_size, block_size),
                        sigma=4.0,
                        theta=orientation,
                        lambd=1.0/frequency if frequency > 0 else 10,
                        gamma=0.5,
                        psi=0
                    )
                    
                    # Apply filter to block
                    block = image[i:i+block_size, j:j+block_size]
                    filtered = cv2.filter2D(block, cv2.CV_32F, kernel)
                    enhanced[i:i+block_size, j:j+block_size] = filtered
        
        # Normalize and convert back to uint8
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        enhanced = enhanced.astype(np.uint8)
        
        # Binarization
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Thinning
        skeleton = cv2.ximgproc.thinning(binary)
        
        return skeleton
    
    def enroll_fingerprint(self, fingerprint_image: np.ndarray, person_id: str) -> bool:
        """Enroll a fingerprint in the database"""
        # Preprocess
        processed = self.preprocess_image(fingerprint_image)
        
        # Quality check
        quality_score, _ = self.quality_assessor.assess_quality(processed)
        if quality_score < 40:
            return False, "Low quality fingerprint"
        
        if self.use_deep_learning:
            # Extract CNN features
            tensor_image = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                features = self.model(tensor_image).numpy()
            
            self.database[person_id] = {
                'type': 'cnn',
                'features': features,
                'quality': quality_score
            }
        else:
            # Extract minutiae
            minutiae = self.extractor.extract_minutiae(processed)
            
            if len(minutiae) < 12:
                return False, "Too few minutiae points"
            
            self.database[person_id] = {
                'type': 'minutiae',
                'minutiae': minutiae,
                'quality': quality_score
            }
        
        return True, "Enrollment successful"
    
    def verify_fingerprint(self, fingerprint_image: np.ndarray, claimed_id: str, 
                          threshold: float = 0.7) -> Tuple[bool, float]:
        """Verify fingerprint against claimed identity"""
        if claimed_id not in self.database:
            return False, 0.0
        
        # Preprocess
        processed = self.preprocess_image(fingerprint_image)
        
        # Quality check
        quality_score, _ = self.quality_assessor.assess_quality(processed)
        if quality_score < 30:
            return False, 0.0
        
        enrolled_data = self.database[claimed_id]
        
        if self.use_deep_learning:
            # CNN verification
            tensor_image = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                features = self.model(tensor_image).numpy()
            
            # Cosine similarity
            similarity = np.dot(features[0], enrolled_data['features'][0]) / (
                np.linalg.norm(features[0]) * np.linalg.norm(enrolled_data['features'][0])
            )
            
            return similarity > threshold, similarity
        else:
            # Minutiae matching
            minutiae = self.extractor.extract_minutiae(processed)
            score = self.matcher.match(minutiae, enrolled_data['minutiae'])
            
            return score > threshold, score
    
    def identify_fingerprint(self, fingerprint_image: np.ndarray, 
                           threshold: float = 0.7) -> Tuple[str, float]:
        """Identify fingerprint from database (1:N matching)"""
        # Preprocess
        processed = self.preprocess_image(fingerprint_image)
        
        # Quality check
        quality_score, _ = self.quality_assessor.assess_quality(processed)
        if quality_score < 30:
            return "Unknown", 0.0
        
        best_match_id = "Unknown"
        best_score = 0.0
        
        # Compare against all enrolled fingerprints
        for person_id, enrolled_data in self.database.items():
            _, score = self.verify_fingerprint(fingerprint_image, person_id, threshold=0)
            
            if score > best_score:
                best_score = score
                best_match_id = person_id
        
        if best_score > threshold:
            return best_match_id, best_score
        else:
            return "Unknown", best_score
    
    def save_database(self, filename: str):
        """Save fingerprint database"""
        with open(filename, 'wb') as f:
            pickle.dump(self.database, f)
    
    def load_database(self, filename: str):
        """Load fingerprint database"""
        with open(filename, 'rb') as f:
            self.database = pickle.load(f)

# Usage example
if __name__ == "__main__":
    # Initialize system
    system = FingerprintRecognitionSystem(use_deep_learning=False)
    
    # Enroll fingerprints
    img1 = cv2.imread("fingerprint1.png", cv2.IMREAD_GRAYSCALE)
    success, message = system.enroll_fingerprint(img1, "person_001")
    print(f"Enrollment: {message}")
    
    # Verify fingerprint
    img2 = cv2.imread("fingerprint2.png", cv2.IMREAD_GRAYSCALE)
    verified, score = system.verify_fingerprint(img2, "person_001")
    print(f"Verification: {'Match' if verified else 'No match'} (score: {score:.3f})")
    
    # Identify fingerprint
    identity, score = system.identify_fingerprint(img2)
    print(f"Identification: {identity} (score: {score:.3f})")
    
    # Save database
    system.save_database("fingerprint_db.pkl")
```

## Mobile Integration

### Android Biometric API
```kotlin
// build.gradle
dependencies {
    implementation 'androidx.biometric:biometric:1.2.0-alpha04'
}

// FingerprintActivity.kt
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import java.util.concurrent.Executor

class FingerprintActivity : AppCompatActivity() {
    private lateinit var executor: Executor
    private lateinit var biometricPrompt: BiometricPrompt
    private lateinit var promptInfo: BiometricPrompt.PromptInfo
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        executor = ContextCompat.getMainExecutor(this)
        
        biometricPrompt = BiometricPrompt(this, executor,
            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                    super.onAuthenticationError(errorCode, errString)
                    Toast.makeText(applicationContext,
                        "Authentication error: $errString", Toast.LENGTH_SHORT).show()
                }
                
                override fun onAuthenticationSucceeded(
                    result: BiometricPrompt.AuthenticationResult) {
                    super.onAuthenticationSucceeded(result)
                    Toast.makeText(applicationContext,
                        "Authentication succeeded!", Toast.LENGTH_SHORT).show()
                    
                    // Process authenticated action
                    processAuthenticatedAction()
                }
                
                override fun onAuthenticationFailed() {
                    super.onAuthenticationFailed()
                    Toast.makeText(applicationContext,
                        "Authentication failed", Toast.LENGTH_SHORT).show()
                }
            })
        
        promptInfo = BiometricPrompt.PromptInfo.Builder()
            .setTitle("Biometric Authentication")
            .setSubtitle("Authenticate using your fingerprint")
            .setNegativeButtonText("Use account password")
            .build()
        
        // Trigger authentication
        findViewById<Button>(R.id.authenticateButton).setOnClickListener {
            biometricPrompt.authenticate(promptInfo)
        }
    }
    
    private fun processAuthenticatedAction() {
        // Perform secure action after authentication
    }
}
```

### iOS Touch ID / Face ID
```swift
import LocalAuthentication

class BiometricAuthManager {
    private let context = LAContext()
    
    func canUseBiometrics() -> Bool {
        var error: NSError?
        return context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, 
                                       error: &error)
    }
    
    func authenticateUser(completion: @escaping (Bool, Error?) -> Void) {
        let reason = "Authenticate to access secure features"
        
        context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics,
                             localizedReason: reason) { success, error in
            DispatchQueue.main.async {
                if success {
                    completion(true, nil)
                } else {
                    completion(false, error)
                }
            }
        }
    }
    
    func getBiometricType() -> String {
        switch context.biometryType {
        case .none:
            return "None"
        case .touchID:
            return "Touch ID"
        case .faceID:
            return "Face ID"
        @unknown default:
            return "Unknown"
        }
    }
}

// Usage
let authManager = BiometricAuthManager()

if authManager.canUseBiometrics() {
    authManager.authenticateUser { success, error in
        if success {
            // Authentication successful
            print("Authenticated successfully")
        } else {
            // Authentication failed
            print("Authentication failed: \(error?.localizedDescription ?? "Unknown error")")
        }
    }
}
```

### React Native Integration
```javascript
// Install react-native-biometrics
// npm install react-native-biometrics

import ReactNativeBiometrics from 'react-native-biometrics'

const rnBiometrics = new ReactNativeBiometrics()

// Check if biometrics are available
rnBiometrics.isSensorAvailable()
  .then((resultObject) => {
    const { available, biometryType } = resultObject
    
    if (available && biometryType === ReactNativeBiometrics.TouchID) {
      console.log('TouchID is supported')
    } else if (available && biometryType === ReactNativeBiometrics.FaceID) {
      console.log('FaceID is supported')
    } else if (available && biometryType === ReactNativeBiometrics.Biometrics) {
      console.log('Biometrics is supported')
    } else {
      console.log('Biometrics not supported')
    }
  })

// Authenticate user
rnBiometrics.simplePrompt({promptMessage: 'Confirm fingerprint'})
  .then((resultObject) => {
    const { success } = resultObject
    
    if (success) {
      console.log('successful biometrics provided')
      // Proceed with authenticated action
    } else {
      console.log('user cancelled biometric prompt')
    }
  })
  .catch(() => {
    console.log('biometrics failed')
  })

// Generate and store keys (for cryptographic operations)
rnBiometrics.createKeys()
  .then((resultObject) => {
    const { publicKey } = resultObject
    console.log(publicKey)
    // Send public key to server
  })

// Sign data with biometric authentication
let payload = 'data to sign'
rnBiometrics.createSignature({
  promptMessage: 'Sign in',
  payload: payload
})
  .then((resultObject) => {
    const { success, signature } = resultObject
    
    if (success) {
      console.log(signature)
      // Send signature to server for verification
    }
  })
```

## Security & Anti-Spoofing

### Liveness Detection
```python
class FingerprintLivenessDetector:
    def __init__(self):
        # Load pre-trained liveness model
        self.model = self._load_liveness_model()
    
    def _load_liveness_model(self):
        """Load CNN model for liveness detection"""
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Live vs Fake
        )
        
        # Load weights
        model.load_state_dict(torch.load('liveness_model.pth'))
        model.eval()
        
        return model
    
    def detect_liveness(self, fingerprint_image):
        """Detect if fingerprint is live or fake"""
        # Preprocess
        processed = cv2.resize(fingerprint_image, (224, 224))
        processed = processed.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor_image = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor_image)
            probabilities = torch.softmax(output, dim=1)
            
            fake_prob = probabilities[0][0].item()
            live_prob = probabilities[0][1].item()
        
        return {
            'is_live': live_prob > fake_prob,
            'live_probability': live_prob,
            'fake_probability': fake_prob
        }
    
    def extract_liveness_features(self, fingerprint_image):
        """Extract features for liveness detection"""
        features = {}
        
        # Texture analysis
        features['texture'] = self._analyze_texture(fingerprint_image)
        
        # Frequency analysis
        features['frequency'] = self._analyze_frequency(fingerprint_image)
        
        # Statistical features
        features['statistics'] = self._compute_statistics(fingerprint_image)
        
        return features
    
    def _analyze_texture(self, image):
        """LBP texture analysis"""
        # Local Binary Patterns
        radius = 3
        n_points = 8 * radius
        
        lbp = cv2.xfeatures2d.LBP_create(radius, n_points)
        hist = lbp.compute(image)[1].flatten()
        
        # Normalize histogram
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def _analyze_frequency(self, image):
        """Frequency domain analysis"""
        # FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Radial frequency distribution
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Binned radial profile
        radial_profile = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())
        
        return radial_profile[:50]  # First 50 frequency bins
    
    def _compute_statistics(self, image):
        """Statistical features"""
        return {
            'mean': np.mean(image),
            'std': np.std(image),
            'skewness': scipy.stats.skew(image.flatten()),
            'kurtosis': scipy.stats.kurtosis(image.flatten()),
            'entropy': -np.sum(image * np.log2(image + 1e-7))
        }
```

### Presentation Attack Detection (PAD)
```python
class PresentationAttackDetector:
    def __init__(self):
        self.attacks = {
            'printed': self._detect_printed_attack,
            'display': self._detect_display_attack,
            'silicone': self._detect_silicone_attack,
            'gelatin': self._detect_gelatin_attack
        }
    
    def detect_attack(self, fingerprint_image, sensor_data=None):
        """Comprehensive attack detection"""
        results = {}
        
        # Image-based detection
        for attack_type, detector in self.attacks.items():
            results[attack_type] = detector(fingerprint_image)
        
        # Sensor-based detection (if available)
        if sensor_data:
            results['sensor_anomaly'] = self._detect_sensor_anomaly(sensor_data)
        
        # Overall decision
        attack_scores = [v['score'] for v in results.values()]
        overall_attack_probability = np.max(attack_scores)
        
        return {
            'is_attack': overall_attack_probability > 0.5,
            'attack_probability': overall_attack_probability,
            'attack_types': results
        }
    
    def _detect_printed_attack(self, image):
        """Detect printed fingerprint attacks"""
        # High frequency noise analysis
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_level = np.std(laplacian)
        
        # Printed images have characteristic noise patterns
        score = 1.0 / (1.0 + np.exp(-0.1 * (noise_level - 50)))
        
        return {'score': score, 'noise_level': noise_level}
    
    def _detect_display_attack(self, image):
        """Detect display-based attacks"""
        # Moiré pattern detection
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Look for regular patterns in frequency domain
        peaks = self._find_frequency_peaks(magnitude)
        
        # Display attacks show regular grid patterns
        score = len(peaks) / 100.0  # Normalize
        
        return {'score': np.clip(score, 0, 1), 'pattern_count': len(peaks)}
    
    def _detect_silicone_attack(self, image):
        """Detect silicone fake fingers"""
        # Ridge clarity analysis
        ridge_clarity = self._compute_ridge_clarity(image)
        
        # Silicone has different optical properties
        score = 1.0 - ridge_clarity
        
        return {'score': score, 'ridge_clarity': ridge_clarity}
    
    def _detect_gelatin_attack(self, image):
        """Detect gelatin fake fingers"""
        # Moisture and conductivity patterns
        local_variance = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Gelatin shows different variance patterns
        score = 1.0 / (1.0 + np.exp(-0.01 * (local_variance - 1000)))
        
        return {'score': score, 'variance': local_variance}
```

## Resources

### Research Papers
- **[Fingerprint Recognition: A Survey](https://ieeexplore.ieee.org/document/8907990)** - IEEE 2020
- **[Deep Learning for Fingerprint Recognition](https://arxiv.org/abs/1907.02395)** - 2019
- **[FingerNet: Unified Deep Network](https://arxiv.org/abs/1709.02228)** - IJCAI 2017
- **[Latent Fingerprint Recognition](https://www.cse.msu.edu/~jain/papers/latent_survey_PIEEE18.pdf)** - MSU

### Books
- "Handbook of Fingerprint Recognition" - Maltoni, Maio, Jain, Prabhakar
- "Biometric Systems: Technology, Design and Performance Evaluation" - Wayman
- "Guide to Biometrics" - Bolle, Connell, Pankanti, Ratha, Senior

### Online Resources
- **[NIST Fingerprint Resources](https://www.nist.gov/programs-projects/fingerprint-recognition)** - Standards and datasets
- **[FVC-onGoing](https://biolab.csr.unibo.it/fvcongoing/)** - Continuous evaluation
- **[Biometrics Research Group](http://biometrics.cse.msu.edu/)** - Michigan State University
- **[OpenBiometrics](http://openbiometrics.org/)** - Open source initiatives

### Competitions & Challenges
- **[FVC-onGoing](https://biolab.csr.unibo.it/fvcongoing/)** - Fingerprint verification
- **[NIST FpVTE](https://www.nist.gov/programs-projects/fingerprint-vendor-technology-evaluation-fpvte)** - Vendor evaluation
- **[LivDet](http://livdet.org/)** - Liveness detection competition