# 👁️ Iris Recognition

Complete guide to iris recognition technology - the most accurate biometric modality with error rates below 0.01%.

**Last Updated:** 2025-06-20

## Table of Contents
- [Introduction](#introduction)
- [Iris Anatomy & Properties](#iris-anatomy--properties)
- [Image Acquisition](#image-acquisition)
- [Iris Segmentation](#iris-segmentation)
- [Feature Extraction](#feature-extraction)
- [Matching Algorithms](#matching-algorithms)
- [Deep Learning Approaches](#deep-learning-approaches)
- [Quality Assessment](#quality-assessment)
- [Frameworks & Libraries](#frameworks--libraries)
- [Datasets & Standards](#datasets--standards)
- [Implementation Guide](#implementation-guide)
- [Challenges & Solutions](#challenges--solutions)
- [Security & Anti-Spoofing](#security--anti-spoofing)
- [Resources](#resources)

## Introduction

Iris recognition uses the unique patterns in the iris (colored ring around the pupil) for identification. It offers:
- **Highest accuracy**: FAR < 0.001%, FRR < 0.1%
- **Stability**: Patterns remain unchanged from 8 months old
- **Non-invasive**: Contactless capture
- **Difficult to spoof**: Protected by cornea
- **Fast matching**: < 100ms for 1:N search

### Applications
- Airport security & border control
- National ID programs (India's Aadhaar)
- High-security facilities
- Banking & financial services
- Healthcare patient identification
- Prison management systems

## Iris Anatomy & Properties

### Unique Features
```python
IRIS_FEATURES = {
    'crypts': 'Diamond-shaped openings',
    'furrows': 'Radial striations',
    'collarette': 'Zigzag boundary',
    'pupillary_zone': 'Inner region',
    'ciliary_zone': 'Outer region',
    'contraction_furrows': 'Concentric lines',
    'pigment_spots': 'Color variations'
}
```

### Mathematical Properties
- **Degrees of freedom**: ~250 independent features
- **Probability of collision**: 1 in 10^78
- **Encoding capacity**: 3.2 bits per square mm
- **Template size**: Typically 512 bytes

## Image Acquisition

### NIR Imaging Systems
```python
class IrisCamera:
    def __init__(self):
        self.wavelength = 850  # nm (Near Infrared)
        self.resolution = (640, 480)
        self.pixel_density = 200  # pixels per iris radius
        self.capture_distance = 30  # cm
        self.illumination_power = 100  # mW/cm²
    
    def configure_camera(self):
        """Configure iris camera settings"""
        settings = {
            'exposure': 'auto',
            'gain': 'auto',
            'focus': 'auto',
            'white_balance': 'disabled',
            'gamma': 1.0,
            'sharpness': 'high',
            'noise_reduction': 'minimal'
        }
        return settings
    
    def capture_iris(self, eye_position):
        """Capture iris image with quality checks"""
        # LED illumination pattern
        illumination = self.create_illumination_pattern()
        
        # Capture multiple frames
        frames = []
        for i in range(5):
            frame = self.capture_frame()
            quality = self.assess_frame_quality(frame)
            
            if quality > 0.8:
                frames.append(frame)
        
        # Select best frame
        best_frame = self.select_best_frame(frames)
        
        return best_frame
```

### Image Quality Requirements
| Parameter | ISO/IEC 19794-6 | NIST IREX |
|-----------|----------------|-----------|
| Resolution | ≥ 150 pixels/radius | ≥ 200 pixels/radius |
| Gray levels | ≥ 8 bits | ≥ 8 bits |
| Image format | Raw, JPEG2000 | Raw, PNG, JPEG2000 |
| Margin | ≥ 70% iris diameter | ≥ 50% iris diameter |
| Contrast | ≥ 50 gray levels | ≥ 80 gray levels |

## Iris Segmentation

### Daugman's Integro-Differential Operator
```python
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class DaugmanSegmentation:
    def __init__(self):
        self.pupil_radius_range = (20, 80)
        self.iris_radius_range = (80, 150)
        self.search_region = 10
    
    def segment_iris(self, eye_image):
        """Segment iris using Daugman's method"""
        # Gaussian smoothing
        smoothed = gaussian_filter(eye_image.astype(float), sigma=1.0)
        
        # Find pupil boundary
        pupil_params = self.find_circle(
            smoothed,
            self.pupil_radius_range,
            search_type='pupil'
        )
        
        # Find iris boundary
        iris_params = self.find_circle(
            smoothed,
            self.iris_radius_range,
            search_type='iris',
            inner_boundary=pupil_params
        )
        
        # Find eyelids
        upper_eyelid = self.detect_eyelid(eye_image, 'upper')
        lower_eyelid = self.detect_eyelid(eye_image, 'lower')
        
        return {
            'pupil': pupil_params,
            'iris': iris_params,
            'upper_eyelid': upper_eyelid,
            'lower_eyelid': lower_eyelid
        }
    
    def find_circle(self, image, radius_range, search_type='iris', inner_boundary=None):
        """Integro-differential operator"""
        max_response = -np.inf
        best_params = None
        
        height, width = image.shape
        
        # Search space
        if inner_boundary and search_type == 'iris':
            # Search around pupil center
            cx_range = range(inner_boundary[0] - self.search_region,
                           inner_boundary[0] + self.search_region)
            cy_range = range(inner_boundary[1] - self.search_region,
                           inner_boundary[1] + self.search_region)
        else:
            # Full image search
            cx_range = range(radius_range[1], width - radius_range[1], 5)
            cy_range = range(radius_range[1], height - radius_range[1], 5)
        
        for cx in cx_range:
            for cy in cy_range:
                for r in range(radius_range[0], radius_range[1], 2):
                    # Compute circular integral
                    response = self.circular_integral(image, cx, cy, r)
                    
                    if response > max_response:
                        max_response = response
                        best_params = (cx, cy, r)
        
        return best_params
    
    def circular_integral(self, image, cx, cy, r):
        """Compute circular integral for given parameters"""
        # Create circular mask
        angles = np.linspace(0, 2*np.pi, 360)
        x = cx + r * np.cos(angles)
        y = cy + r * np.sin(angles)
        
        # Ensure within bounds
        mask = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        x, y = x[mask].astype(int), y[mask].astype(int)
        
        if len(x) == 0:
            return -np.inf
        
        # Compute gradient along circle
        values = image[y, x]
        gradient = np.gradient(values)
        
        # Response is sum of gradients
        return np.sum(np.abs(gradient))
```

### Deep Learning Segmentation
```python
import torch
import torch.nn as nn

class IrisUNet(nn.Module):
    """U-Net for iris segmentation"""
    def __init__(self, in_channels=1, out_channels=4):
        super(IrisUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer (4 classes: background, iris, pupil, eyelids)
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([e4, self.upconv4(b)], 1))
        d3 = self.dec3(torch.cat([e3, self.upconv3(d4)], 1))
        d2 = self.dec2(torch.cat([e2, self.upconv2(d3)], 1))
        d1 = self.dec1(torch.cat([e1, self.upconv1(d2)], 1))
        
        return self.out(d1)
```

## Feature Extraction

### Daugman's IrisCode
```python
class IrisCode:
    def __init__(self, n_scales=4, n_orientations=4):
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.angular_resolution = 256
        self.radial_resolution = 32
    
    def encode_iris(self, iris_image, segmentation):
        """Generate IrisCode using 2D Gabor wavelets"""
        # Normalize iris to rectangular representation
        normalized = self.normalize_iris(iris_image, segmentation)
        
        # Create Gabor filter bank
        gabor_bank = self.create_gabor_bank()
        
        # Apply filters and encode
        iris_code = []
        
        for scale in range(self.n_scales):
            for orientation in range(self.n_orientations):
                # Apply Gabor filter
                filtered = cv2.filter2D(normalized, cv2.CV_64F, gabor_bank[scale][orientation])
                
                # Phase quantization
                real_part = np.real(filtered)
                imag_part = np.imag(filtered)
                
                # 2-bit encoding per pixel
                code = np.zeros_like(real_part, dtype=np.uint8)
                code[real_part > 0] |= 1
                code[imag_part > 0] |= 2
                
                iris_code.append(code)
        
        # Flatten to binary template
        iris_code = np.concatenate([c.flatten() for c in iris_code])
        
        # Create mask for valid regions
        mask = self.create_mask(normalized, segmentation)
        
        return iris_code, mask
    
    def normalize_iris(self, image, segmentation):
        """Rubber sheet normalization"""
        pupil_x, pupil_y, pupil_r = segmentation['pupil']
        iris_x, iris_y, iris_r = segmentation['iris']
        
        # Create normalized coordinates
        theta = np.linspace(0, 2*np.pi, self.angular_resolution)
        r = np.linspace(0, 1, self.radial_resolution)
        
        normalized = np.zeros((self.radial_resolution, self.angular_resolution))
        
        for i, r_val in enumerate(r):
            for j, theta_val in enumerate(theta):
                # Map from normalized to Cartesian
                x_pupil = pupil_x + pupil_r * np.cos(theta_val)
                y_pupil = pupil_y + pupil_r * np.sin(theta_val)
                
                x_iris = iris_x + iris_r * np.cos(theta_val)
                y_iris = iris_y + iris_r * np.sin(theta_val)
                
                # Linear interpolation
                x = (1 - r_val) * x_pupil + r_val * x_iris
                y = (1 - r_val) * y_pupil + r_val * y_iris
                
                # Bilinear interpolation
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    normalized[i, j] = self.bilinear_interpolate(image, x, y)
        
        return normalized
    
    def create_gabor_bank(self):
        """Create 2D Gabor filter bank"""
        gabor_bank = []
        
        for scale in range(self.n_scales):
            scale_filters = []
            wavelength = 2 ** (scale + 2)
            
            for orientation in range(self.n_orientations):
                theta = orientation * np.pi / self.n_orientations
                
                # Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(17, 17),
                    sigma=wavelength / 2,
                    theta=theta,
                    lambd=wavelength,
                    gamma=0.5,
                    psi=0
                )
                
                scale_filters.append(kernel)
            
            gabor_bank.append(scale_filters)
        
        return gabor_bank
    
    def hamming_distance(self, code1, mask1, code2, mask2):
        """Fractional Hamming distance with masking"""
        # Only compare valid bits
        valid_mask = mask1 & mask2
        
        if np.sum(valid_mask) == 0:
            return 1.0  # No valid comparison
        
        # XOR codes
        diff = (code1 ^ code2) & valid_mask
        
        # Fractional HD
        hd = np.sum(diff) / np.sum(valid_mask)
        
        return hd
```

### Deep Learning Features
```python
class IrisNet(nn.Module):
    """Deep CNN for iris feature extraction"""
    def __init__(self, embedding_size=512):
        super(IrisNet, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_size)
        )
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Generate embedding
        embedding = self.embedding(x)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

class TripletLoss(nn.Module):
    """Triplet loss for iris recognition"""
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()
```

## Matching Algorithms

### Hamming Distance Matching
```python
class IrisMatcher:
    def __init__(self, threshold=0.32):
        self.threshold = threshold  # Typical threshold for IrisCode
        self.rotation_range = (-10, 10)  # degrees
    
    def match_iriscodes(self, code1, mask1, code2, mask2):
        """Match IrisCodes with rotation compensation"""
        min_hd = 1.0
        best_shift = 0
        
        # Angular resolution
        angular_res = len(code1) // (32 * 4 * 4)  # Based on encoding params
        
        # Try different rotations
        for shift in range(self.rotation_range[0], self.rotation_range[1]):
            # Circular shift in angular dimension
            shifted_code2 = self.circular_shift_code(code2, shift, angular_res)
            shifted_mask2 = self.circular_shift_code(mask2, shift, angular_res)
            
            # Compute Hamming distance
            hd = self.fractional_hamming_distance(
                code1, mask1, shifted_code2, shifted_mask2
            )
            
            if hd < min_hd:
                min_hd = hd
                best_shift = shift
        
        return {
            'distance': min_hd,
            'match': min_hd < self.threshold,
            'rotation': best_shift,
            'confidence': 1.0 - min_hd
        }
    
    def fractional_hamming_distance(self, code1, mask1, code2, mask2):
        """Compute fractional Hamming distance"""
        # Valid bits are where both masks are 1
        valid_bits = mask1 & mask2
        n_valid = np.sum(valid_bits)
        
        if n_valid == 0:
            return 1.0
        
        # XOR codes where valid
        disagreements = (code1 ^ code2) & valid_bits
        n_disagreements = np.sum(disagreements)
        
        return n_disagreements / n_valid
    
    def circular_shift_code(self, code, shift, angular_res):
        """Circular shift for rotation compensation"""
        # Reshape to 2D (radial x angular)
        radial_res = len(code) // angular_res
        code_2d = code.reshape(radial_res, angular_res)
        
        # Circular shift along angular dimension
        shifted = np.roll(code_2d, shift, axis=1)
        
        return shifted.flatten()
```

### Score Normalization
```python
class ScoreNormalizer:
    def __init__(self):
        self.methods = {
            'min_max': self.min_max_norm,
            'z_score': self.z_score_norm,
            'tanh': self.tanh_norm,
            'adaptive': self.adaptive_norm
        }
    
    def normalize_scores(self, scores, method='adaptive'):
        """Normalize matching scores"""
        return self.methods[method](scores)
    
    def min_max_norm(self, scores):
        """Min-max normalization"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_score) / (max_score - min_score)
    
    def z_score_norm(self, scores):
        """Z-score normalization"""
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return np.zeros_like(scores)
        
        return (scores - mean) / std
    
    def tanh_norm(self, scores):
        """Hyperbolic tangent normalization"""
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return np.ones_like(scores) * 0.5
        
        normalized = 0.5 * (np.tanh(0.01 * (scores - mean) / std) + 1)
        return normalized
    
    def adaptive_norm(self, scores):
        """Adaptive normalization based on score distribution"""
        # Estimate genuine and impostor distributions
        hist, bins = np.histogram(scores, bins=50)
        
        # Find valley between distributions
        valley_idx = self.find_valley(hist)
        threshold = bins[valley_idx]
        
        # Separate scores
        genuine_scores = scores[scores > threshold]
        impostor_scores = scores[scores <= threshold]
        
        # Normalize separately
        normalized = np.zeros_like(scores)
        
        if len(genuine_scores) > 0:
            genuine_norm = self.min_max_norm(genuine_scores)
            normalized[scores > threshold] = 0.5 + 0.5 * genuine_norm
        
        if len(impostor_scores) > 0:
            impostor_norm = self.min_max_norm(impostor_scores)
            normalized[scores <= threshold] = 0.5 * impostor_norm
        
        return normalized
    
    def find_valley(self, histogram):
        """Find valley in bimodal distribution"""
        # Smooth histogram
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(histogram, sigma=2)
        
        # Find local minima
        minima = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                minima.append(i)
        
        # Return middle minimum or center if none found
        if minima:
            return minima[len(minima)//2]
        else:
            return len(histogram) // 2
```

## Deep Learning Approaches

### UniNet - Unified Network for Iris
```python
class UniNet(nn.Module):
    """Unified network for segmentation, normalization, and matching"""
    def __init__(self):
        super(UniNet, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Segmentation branch
        self.seg_branch = IrisUNet(64, 4)
        
        # Normalization branch
        self.norm_branch = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction branch
        self.feature_branch = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        # Shared features
        shared = self.backbone(x)
        
        # Segmentation
        seg_mask = self.seg_branch(shared)
        
        # Apply segmentation mask
        masked_features = shared * torch.sigmoid(seg_mask[:, 1:2])  # Iris mask
        
        # Normalization
        normalized = self.norm_branch(masked_features)
        
        # Feature extraction
        features = self.feature_branch(normalized)
        
        return {
            'segmentation': seg_mask,
            'features': features
        }
```

### Cross-Sensor Recognition
```python
class CrossSensorIrisNet(nn.Module):
    """Domain adaptation for cross-sensor iris recognition"""
    def __init__(self, n_sensors=3):
        super(CrossSensorIrisNet, self).__init__()
        
        # Sensor-specific encoders
        self.encoders = nn.ModuleList([
            self._make_encoder() for _ in range(n_sensors)
        ])
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_sensors)
        )
        
        # Identity classifier
        self.identity_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)  # Embedding
        )
    
    def _make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, sensor_id=None, alpha=1.0):
        # Sensor-specific encoding
        if sensor_id is not None:
            encoded = self.encoders[sensor_id](x)
        else:
            # Average all encoders (inference)
            encoded = sum(encoder(x) for encoder in self.encoders) / len(self.encoders)
        
        # Shared features
        features = self.shared_features(encoded)
        features = features.view(features.size(0), -1)
        
        # Identity embedding
        identity = self.identity_classifier(features)
        
        # Domain prediction (with gradient reversal)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain = self.domain_classifier(reversed_features)
        
        return identity, domain

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

## Quality Assessment

### ISO/IEC 29794-6 Quality Metrics
```python
class IrisQualityAssessment:
    def __init__(self):
        self.metrics = {
            'usable_area': self.compute_usable_area,
            'iris_sclera_contrast': self.compute_iris_sclera_contrast,
            'iris_pupil_contrast': self.compute_iris_pupil_contrast,
            'pupil_boundary_circularity': self.compute_pupil_circularity,
            'gray_scale_utilization': self.compute_gray_scale_utilization,
            'sharpness': self.compute_sharpness,
            'motion_blur': self.compute_motion_blur,
            'signal_to_noise': self.compute_snr
        }
    
    def assess_quality(self, iris_image, segmentation):
        """Comprehensive iris quality assessment"""
        scores = {}
        
        for metric_name, metric_func in self.metrics.items():
            scores[metric_name] = metric_func(iris_image, segmentation)
        
        # Overall quality score (0-100)
        overall_score = self.compute_overall_score(scores)
        
        return overall_score, scores
    
    def compute_usable_area(self, image, segmentation):
        """Percentage of iris not occluded"""
        iris_mask = self.create_iris_mask(image.shape, segmentation)
        
        # Detect occlusions (eyelids, eyelashes, reflections)
        occlusions = self.detect_occlusions(image, segmentation)
        
        # Usable area ratio
        total_iris_pixels = np.sum(iris_mask)
        occluded_pixels = np.sum(occlusions & iris_mask)
        
        usable_ratio = (total_iris_pixels - occluded_pixels) / total_iris_pixels
        
        return usable_ratio
    
    def compute_iris_sclera_contrast(self, image, segmentation):
        """Contrast between iris and sclera"""
        iris_x, iris_y, iris_r = segmentation['iris']
        
        # Extract iris region
        iris_mask = self.create_circular_mask(image.shape, iris_x, iris_y, iris_r)
        iris_pixels = image[iris_mask]
        
        # Extract sclera region (outside iris)
        sclera_mask = self.create_annular_mask(
            image.shape, iris_x, iris_y, iris_r, iris_r + 20
        )
        sclera_pixels = image[sclera_mask]
        
        if len(iris_pixels) == 0 or len(sclera_pixels) == 0:
            return 0.0
        
        # Contrast metric
        contrast = abs(np.mean(iris_pixels) - np.mean(sclera_pixels))
        
        return contrast / 255.0
    
    def compute_iris_pupil_contrast(self, image, segmentation):
        """Contrast between iris and pupil"""
        pupil_x, pupil_y, pupil_r = segmentation['pupil']
        iris_x, iris_y, iris_r = segmentation['iris']
        
        # Extract regions
        pupil_mask = self.create_circular_mask(image.shape, pupil_x, pupil_y, pupil_r)
        iris_mask = self.create_annular_mask(
            image.shape, iris_x, iris_y, pupil_r, iris_r
        )
        
        pupil_pixels = image[pupil_mask]
        iris_pixels = image[iris_mask]
        
        if len(pupil_pixels) == 0 or len(iris_pixels) == 0:
            return 0.0
        
        # Contrast metric
        contrast = abs(np.mean(iris_pixels) - np.mean(pupil_pixels))
        
        return contrast / 255.0
    
    def compute_pupil_circularity(self, image, segmentation):
        """Measure how circular the pupil is"""
        pupil_x, pupil_y, pupil_r = segmentation['pupil']
        
        # Extract pupil boundary
        pupil_edge = self.extract_boundary(image, pupil_x, pupil_y, pupil_r)
        
        # Fit ellipse
        if len(pupil_edge) < 5:
            return 0.0
        
        ellipse = cv2.fitEllipse(pupil_edge)
        (cx, cy), (major_axis, minor_axis), angle = ellipse
        
        # Circularity = minor/major axis ratio
        circularity = minor_axis / major_axis if major_axis > 0 else 0
        
        return circularity
    
    def compute_gray_scale_utilization(self, image, segmentation):
        """Dynamic range utilization"""
        iris_mask = self.create_iris_mask(image.shape, segmentation)
        iris_pixels = image[iris_mask]
        
        if len(iris_pixels) == 0:
            return 0.0
        
        # Check histogram spread
        hist, _ = np.histogram(iris_pixels, bins=256, range=(0, 255))
        
        # Find effective range
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        
        # 1% and 99% percentiles
        low_idx = np.searchsorted(cumsum, 0.01 * total)
        high_idx = np.searchsorted(cumsum, 0.99 * total)
        
        # Utilization score
        utilization = (high_idx - low_idx) / 256.0
        
        return utilization
    
    def compute_sharpness(self, image, segmentation):
        """Iris texture sharpness"""
        iris_mask = self.create_iris_mask(image.shape, segmentation)
        
        # Laplacian variance method
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Only consider iris region
        iris_laplacian = laplacian[iris_mask]
        
        if len(iris_laplacian) == 0:
            return 0.0
        
        # Variance as sharpness measure
        sharpness = np.var(iris_laplacian)
        
        # Normalize to 0-1 range
        normalized_sharpness = 1.0 - np.exp(-sharpness / 1000)
        
        return normalized_sharpness
    
    def compute_motion_blur(self, image, segmentation):
        """Detect motion blur in iris"""
        iris_mask = self.create_iris_mask(image.shape, segmentation)
        
        # Frequency domain analysis
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Motion blur creates directional patterns in frequency domain
        # Compute directional energy
        angles = np.linspace(0, np.pi, 180)
        energy_distribution = []
        
        for angle in angles:
            # Create directional filter
            rows, cols = image.shape
            center = (rows // 2, cols // 2)
            
            # Sample along direction
            samples = []
            for r in range(min(rows, cols) // 4):
                x = int(center[1] + r * np.cos(angle))
                y = int(center[0] + r * np.sin(angle))
                
                if 0 <= x < cols and 0 <= y < rows:
                    samples.append(magnitude[y, x])
            
            if samples:
                energy_distribution.append(np.mean(samples))
        
        # Motion blur shows high variance in directional energy
        blur_score = np.std(energy_distribution) / (np.mean(energy_distribution) + 1e-7)
        
        # Invert and normalize (0=blur, 1=sharp)
        quality = 1.0 / (1.0 + blur_score)
        
        return quality
    
    def compute_snr(self, image, segmentation):
        """Signal-to-noise ratio"""
        iris_mask = self.create_iris_mask(image.shape, segmentation)
        iris_region = image[iris_mask]
        
        if len(iris_region) == 0:
            return 0.0
        
        # Estimate signal (smooth component)
        smooth = cv2.GaussianBlur(image, (5, 5), 1.0)
        signal = smooth[iris_mask]
        
        # Estimate noise
        noise = iris_region - signal
        
        # SNR in dB
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return 1.0
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Normalize to 0-1 range
        normalized_snr = 1.0 / (1.0 + np.exp(-0.1 * (snr_db - 20)))
        
        return normalized_snr
    
    def compute_overall_score(self, individual_scores):
        """Combine individual metrics into overall score"""
        # Weights based on importance
        weights = {
            'usable_area': 0.20,
            'iris_sclera_contrast': 0.10,
            'iris_pupil_contrast': 0.10,
            'pupil_boundary_circularity': 0.10,
            'gray_scale_utilization': 0.10,
            'sharpness': 0.20,
            'motion_blur': 0.10,
            'signal_to_noise': 0.10
        }
        
        # Weighted sum
        overall = sum(weights[k] * individual_scores[k] for k in weights)
        
        # Convert to 0-100 scale
        return int(overall * 100)
```

## Frameworks & Libraries

### Open Source Libraries

#### USIT - University of Salzburg Iris Toolkit
```python
# Installation
# git clone https://github.com/USIT/USIT
# cd USIT && python setup.py install

from USIT import iris_recognition

# Load iris image
iris = iris_recognition.load_iris_image("iris.png")

# Segment iris
segmentation = iris_recognition.segment_iris(iris)

# Extract features
features = iris_recognition.extract_features(iris, segmentation)

# Match irises
score = iris_recognition.match_features(features1, features2)
```

#### PyIris
```python
# Simple iris recognition library
import pyiris

# Initialize system
iris_system = pyiris.IrisRecognitionSystem()

# Enroll iris
iris_system.enroll("person_001", "iris_image.png")

# Identify
identity, confidence = iris_system.identify("query_iris.png")
```

#### OpenCV Iris Recognition
```python
import cv2
import numpy as np

class OpenCVIris:
    def __init__(self):
        # Haar cascade for eye detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_iris(self, image):
        """Basic iris detection using HoughCircles"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes first
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (ex, ey, ew, eh) in eyes:
            eye_region = gray[ey:ey+eh, ex:ex+ew]
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(eye_region, (5, 5), 0)
            
            # Detect circles (pupil and iris)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=80
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                # Sort by radius (pupil < iris)
                sorted_circles = sorted(circles[0], key=lambda x: x[2])
                
                if len(sorted_circles) >= 2:
                    pupil = sorted_circles[0]
                    iris = sorted_circles[1]
                    
                    return {
                        'pupil': (pupil[0] + ex, pupil[1] + ey, pupil[2]),
                        'iris': (iris[0] + ex, iris[1] + ey, iris[2])
                    }
        
        return None
```

### Commercial SDKs

#### Neurotechnology VeriEye
```python
# VeriEye SDK example
from neurotec.biometrics import NIris, NSubject
from neurotec.biometrics.client import NBiometricClient

class VeriEyeSDK:
    def __init__(self):
        self.client = NBiometricClient()
        self.client.iris_matching_speed = "high"
    
    def extract_template(self, image_path):
        """Extract iris template"""
        subject = NSubject()
        iris = NIris()
        iris.image = NImage.from_file(image_path)
        subject.irises.add(iris)
        
        # Extract template
        status = self.client.create_template(subject)
        
        if status == NBiometricStatus.OK:
            return subject.template
        else:
            raise Exception(f"Template extraction failed: {status}")
    
    def match_templates(self, template1, template2):
        """Match two iris templates"""
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

#### IriShield SDK
```python
# IriShield MK 2120U integration
import irishield

# Initialize device
device = irishield.IriShieldDevice()
device.connect()

# Capture iris
left_iris, right_iris = device.capture_both_eyes()

# Quality check
if left_iris.quality > 80:
    # Extract template
    template = irishield.extract_template(left_iris.image)
    
    # Save template
    irishield.save_template(template, "user_001_left.iri")

# Match templates
score = irishield.match_templates("template1.iri", "template2.iri")
print(f"Match score: {score}")
```

## Datasets & Standards

### Public Datasets

#### CASIA Iris Databases
| Dataset | Images | Subjects | Sensors | Resolution | Notes |
|---------|--------|----------|---------|------------|-------|
| **[CASIA-IrisV1](https://biometrics.idealtest.org/)** | 756 | 108 | 1 | 320×280 | Close-up |
| **[CASIA-IrisV3](https://biometrics.idealtest.org/)** | 22,051 | 700+ | 3 | 640×480 | Interval, Lamp, Twins |
| **[CASIA-IrisV4](https://biometrics.idealtest.org/)** | 54,607 | 1,800 | Multiple | Various | Distance, Thousand, Syn |
| **[CASIA-Iris-Mobile](https://biometrics.idealtest.org/)** | 12,000 | 630 | Mobile | 640×480 | Mobile devices |

#### Other Datasets
- **[UBIRIS.v1](https://iris.di.ubi.pt/)** - Visible light, noisy conditions
- **[UBIRIS.v2](https://iris.di.ubi.pt/)** - Moving subjects, realistic
- **[ND-IRIS-0405](https://cvrl.nd.edu/)** - Notre Dame dataset
- **[IITD Iris](https://sites.google.com/site/drkumariitkgp/iris-database)** - IIT Delhi database
- **[MMU Iris](https://www.cs.princeton.edu/~andyz/iris_database.html)** - Multimedia University

### Standards

#### ISO/IEC 19794-6
```python
class ISO19794_6:
    """ISO/IEC 19794-6:2011 Iris image data format"""
    
    def __init__(self):
        self.format_identifier = b'IIR\x00'
        self.version = b'030\x00'  # Version 3.0
        self.capture_device_id = 0
        self.horizontal_orientation = 0  # 0=undefined, 1=base, 2=flipped
        self.vertical_orientation = 0
        self.image_format = 2  # 2=raw, 14=PNG, 15=JPEG2000
    
    def encode_record(self, iris_images):
        """Encode iris images to ISO format"""
        record = bytearray()
        
        # General header
        record.extend(self.format_identifier)
        record.extend(self.version)
        record.extend(struct.pack('>I', 0))  # Record length (update later)
        record.extend(struct.pack('>H', self.capture_device_id))
        record.append(len(iris_images))  # Number of images
        
        # Image records
        for img_data in iris_images:
            img_record = self.encode_image_record(img_data)
            record.extend(img_record)
        
        # Update record length
        record_length = len(record)
        record[8:12] = struct.pack('>I', record_length)
        
        return bytes(record)
    
    def encode_image_record(self, img_data):
        """Encode single iris image"""
        record = bytearray()
        
        # Image properties
        record.append(img_data['eye_label'])  # 1=right, 2=left
        record.append(img_data['image_type'])  # 1=uncropped, 2=cropped, 7=cropped_masked
        record.extend(struct.pack('>H', img_data['width']))
        record.extend(struct.pack('>H', img_data['height']))
        record.append(img_data['bit_depth'])  # Usually 8
        record.extend(struct.pack('>H', img_data['range']))  # e.g., 256 for 8-bit
        record.extend(struct.pack('>H', self.horizontal_orientation))
        record.extend(struct.pack('>H', self.vertical_orientation))
        record.append(self.image_format)
        
        # Image data
        image_bytes = img_data['data']
        record.extend(struct.pack('>I', len(image_bytes)))
        record.extend(image_bytes)
        
        return record
```

#### ANSI/NIST-ITL Type-17
```python
class ANSINISTType17:
    """ANSI/NIST-ITL 1-2011 Type-17 Iris Image"""
    
    def __init__(self):
        self.fields = {
            '17.001': 'RECORD HEADER',
            '17.002': 'INFORMATION DESIGNATION CHARACTER',
            '17.003': 'EYE LABEL',
            '17.004': 'SOURCE AGENCY',
            '17.005': 'IRIS CAPTURE DATE',
            '17.006': 'HORIZONTAL LINE LENGTH',
            '17.007': 'VERTICAL LINE LENGTH',
            '17.008': 'SCALE UNITS',
            '17.009': 'TRANSMITTED HORIZONTAL PIXEL SCALE',
            '17.010': 'TRANSMITTED VERTICAL PIXEL SCALE',
            '17.011': 'COMPRESSION ALGORITHM',
            '17.012': 'BITS PER PIXEL',
            '17.013': 'COLOR SPACE',
            '17.014': 'ROTATION ANGLE',
            '17.015': 'ROTATION UNCERTAINTY',
            '17.999': 'IMAGE DATA'
        }
    
    def create_record(self, iris_image, metadata):
        """Create Type-17 record"""
        record = {}
        
        # Mandatory fields
        record['17.001'] = self._create_header()
        record['17.002'] = 'IDC'
        record['17.003'] = metadata.get('eye_label', '0')  # 0=unknown, 1=right, 2=left
        record['17.004'] = metadata.get('source_agency', 'UNKNOWN')
        record['17.005'] = metadata.get('capture_date', '20240101')
        record['17.006'] = str(iris_image.shape[1])  # Width
        record['17.007'] = str(iris_image.shape[0])  # Height
        record['17.008'] = '1'  # 1=pixels/inch, 2=pixels/cm
        record['17.009'] = '500'  # Horizontal resolution
        record['17.010'] = '500'  # Vertical resolution
        record['17.011'] = 'NONE'  # Compression
        record['17.012'] = '8'  # Bits per pixel
        record['17.013'] = 'GRAY'  # Color space
        
        # Optional fields
        if 'rotation_angle' in metadata:
            record['17.014'] = str(metadata['rotation_angle'])
            record['17.015'] = str(metadata.get('rotation_uncertainty', 2))
        
        # Image data
        record['17.999'] = self._encode_image(iris_image)
        
        return record
    
    def _create_header(self):
        """Create record header"""
        # Implementation details...
        return "17.001:TYPE-17"
    
    def _encode_image(self, image):
        """Encode image data"""
        # Convert to bytes
        return image.tobytes()
```

## Implementation Guide

### Complete Iris Recognition System
```python
import cv2
import numpy as np
import torch
from typing import Dict, Tuple, List
import json

class IrisRecognitionSystem:
    def __init__(self, config_path='iris_config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.segmenter = IrisSegmenter(self.config['segmentation'])
        self.encoder = IrisEncoder(self.config['encoding'])
        self.matcher = IrisMatcher(self.config['matching'])
        self.quality_assessor = IrisQualityAssessment()
        
        # Database
        self.database = {}
        
        # Load deep learning models if enabled
        if self.config.get('use_deep_learning', False):
            self.load_deep_models()
    
    def load_deep_models(self):
        """Load pre-trained deep learning models"""
        # Segmentation model
        self.seg_model = IrisUNet()
        self.seg_model.load_state_dict(
            torch.load(self.config['models']['segmentation'])
        )
        self.seg_model.eval()
        
        # Feature extraction model
        self.feature_model = IrisNet()
        self.feature_model.load_state_dict(
            torch.load(self.config['models']['features'])
        )
        self.feature_model.eval()
    
    def enroll_iris(self, image_path: str, person_id: str, 
                    eye_label: str = 'unknown') -> Dict:
        """Enroll iris in database"""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {'success': False, 'error': 'Failed to load image'}
        
        # Quality assessment
        quality_score, quality_details = self.assess_image_quality(image)
        
        if quality_score < self.config['quality_threshold']:
            return {
                'success': False,
                'error': 'Low quality image',
                'quality_score': quality_score,
                'details': quality_details
            }
        
        # Segment iris
        segmentation = self.segment_iris(image)
        
        if segmentation is None:
            return {'success': False, 'error': 'Segmentation failed'}
        
        # Extract features
        if self.config.get('use_deep_learning', False):
            features = self.extract_deep_features(image, segmentation)
        else:
            features = self.extract_traditional_features(image, segmentation)
        
        # Store in database
        if person_id not in self.database:
            self.database[person_id] = {}
        
        self.database[person_id][eye_label] = {
            'features': features,
            'quality': quality_score,
            'segmentation': segmentation,
            'metadata': {
                'enrollment_date': str(datetime.now()),
                'image_path': image_path
            }
        }
        
        return {
            'success': True,
            'person_id': person_id,
            'eye_label': eye_label,
            'quality_score': quality_score
        }
    
    def verify_iris(self, image_path: str, claimed_id: str, 
                   eye_label: str = 'unknown') -> Dict:
        """Verify iris against claimed identity"""
        # Check if person exists
        if claimed_id not in self.database:
            return {'verified': False, 'error': 'Person not enrolled'}
        
        if eye_label != 'unknown' and eye_label not in self.database[claimed_id]:
            return {'verified': False, 'error': f'{eye_label} eye not enrolled'}
        
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {'verified': False, 'error': 'Failed to load image'}
        
        # Quality check
        quality_score, _ = self.assess_image_quality(image)
        
        if quality_score < self.config['quality_threshold'] * 0.8:  # Lower threshold for verification
            return {
                'verified': False,
                'error': 'Low quality image',
                'quality_score': quality_score
            }
        
        # Segment and extract features
        segmentation = self.segment_iris(image)
        
        if segmentation is None:
            return {'verified': False, 'error': 'Segmentation failed'}
        
        if self.config.get('use_deep_learning', False):
            query_features = self.extract_deep_features(image, segmentation)
        else:
            query_features = self.extract_traditional_features(image, segmentation)
        
        # Match against enrolled eye(s)
        if eye_label == 'unknown':
            # Try both eyes
            eyes_to_check = list(self.database[claimed_id].keys())
        else:
            eyes_to_check = [eye_label]
        
        best_score = 0
        best_eye = None
        
        for eye in eyes_to_check:
            enrolled_data = self.database[claimed_id][eye]
            score = self.match_features(
                query_features,
                enrolled_data['features']
            )
            
            if score > best_score:
                best_score = score
                best_eye = eye
        
        # Decision
        verified = best_score > self.config['verification_threshold']
        
        return {
            'verified': verified,
            'score': best_score,
            'eye_matched': best_eye,
            'confidence': self.score_to_confidence(best_score)
        }
    
    def identify_iris(self, image_path: str) -> Dict:
        """Identify iris from database (1:N matching)"""
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {'identified': False, 'error': 'Failed to load image'}
        
        # Quality check
        quality_score, _ = self.assess_image_quality(image)
        
        if quality_score < self.config['quality_threshold'] * 0.8:
            return {
                'identified': False,
                'error': 'Low quality image',
                'quality_score': quality_score
            }
        
        # Segment and extract features
        segmentation = self.segment_iris(image)
        
        if segmentation is None:
            return {'identified': False, 'error': 'Segmentation failed'}
        
        if self.config.get('use_deep_learning', False):
            query_features = self.extract_deep_features(image, segmentation)
        else:
            query_features = self.extract_traditional_features(image, segmentation)
        
        # Search database
        best_score = 0
        best_match = None
        best_eye = None
        
        for person_id, person_data in self.database.items():
            for eye_label, eye_data in person_data.items():
                score = self.match_features(
                    query_features,
                    eye_data['features']
                )
                
                if score > best_score:
                    best_score = score
                    best_match = person_id
                    best_eye = eye_label
        
        # Decision
        identified = best_score > self.config['identification_threshold']
        
        return {
            'identified': identified,
            'person_id': best_match if identified else None,
            'eye_label': best_eye if identified else None,
            'score': best_score,
            'confidence': self.score_to_confidence(best_score)
        }
    
    def segment_iris(self, image: np.ndarray) -> Dict:
        """Segment iris using configured method"""
        if self.config.get('use_deep_learning', False):
            # Deep learning segmentation
            tensor_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            
            with torch.no_grad():
                seg_output = self.seg_model(tensor_image)
                seg_masks = torch.argmax(seg_output, dim=1).squeeze().numpy()
            
            # Extract parameters from masks
            segmentation = self.masks_to_parameters(seg_masks)
        else:
            # Traditional segmentation
            segmentation = self.segmenter.segment_iris(image)
        
        return segmentation
    
    def extract_traditional_features(self, image: np.ndarray, 
                                   segmentation: Dict) -> Dict:
        """Extract IrisCode features"""
        iris_code, mask = self.encoder.encode_iris(image, segmentation)
        
        return {
            'type': 'iriscode',
            'code': iris_code,
            'mask': mask,
            'segmentation': segmentation
        }
    
    def extract_deep_features(self, image: np.ndarray, 
                            segmentation: Dict) -> Dict:
        """Extract deep learning features"""
        # Normalize iris
        normalized = self.encoder.normalize_iris(image, segmentation)
        
        # Convert to tensor
        tensor_norm = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
        
        # Extract features
        with torch.no_grad():
            features = self.feature_model(tensor_norm).squeeze().numpy()
        
        return {
            'type': 'deep',
            'embedding': features,
            'segmentation': segmentation
        }
    
    def match_features(self, features1: Dict, features2: Dict) -> float:
        """Match two feature sets"""
        if features1['type'] != features2['type']:
            raise ValueError("Feature types must match")
        
        if features1['type'] == 'iriscode':
            # Hamming distance matching
            score = 1.0 - self.matcher.match_iriscodes(
                features1['code'], features1['mask'],
                features2['code'], features2['mask']
            )['distance']
        else:
            # Cosine similarity for deep features
            embedding1 = features1['embedding']
            embedding2 = features2['embedding']
            
            score = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        
        return score
    
    def assess_image_quality(self, image: np.ndarray) -> Tuple[float, Dict]:
        """Assess iris image quality"""
        # Quick segmentation for quality assessment
        segmentation = self.segmenter.segment_iris(image)
        
        if segmentation is None:
            return 0.0, {'error': 'Segmentation failed'}
        
        return self.quality_assessor.assess_quality(image, segmentation)
    
    def score_to_confidence(self, score: float) -> str:
        """Convert matching score to confidence level"""
        if score > 0.95:
            return 'Very High'
        elif score > 0.85:
            return 'High'
        elif score > 0.75:
            return 'Medium'
        elif score > 0.65:
            return 'Low'
        else:
            return 'Very Low'
    
    def save_database(self, filepath: str):
        """Save database to file"""
        # Convert numpy arrays to lists for JSON serialization
        db_serializable = {}
        
        for person_id, person_data in self.database.items():
            db_serializable[person_id] = {}
            
            for eye_label, eye_data in person_data.items():
                features = eye_data['features'].copy()
                
                # Convert numpy arrays
                if features['type'] == 'iriscode':
                    features['code'] = features['code'].tolist()
                    features['mask'] = features['mask'].tolist()
                else:
                    features['embedding'] = features['embedding'].tolist()
                
                db_serializable[person_id][eye_label] = {
                    'features': features,
                    'quality': eye_data['quality'],
                    'metadata': eye_data['metadata']
                }
        
        with open(filepath, 'w') as f:
            json.dump(db_serializable, f, indent=2)
    
    def load_database(self, filepath: str):
        """Load database from file"""
        with open(filepath, 'r') as f:
            db_loaded = json.load(f)
        
        # Convert lists back to numpy arrays
        self.database = {}
        
        for person_id, person_data in db_loaded.items():
            self.database[person_id] = {}
            
            for eye_label, eye_data in person_data.items():
                features = eye_data['features'].copy()
                
                # Convert to numpy
                if features['type'] == 'iriscode':
                    features['code'] = np.array(features['code'])
                    features['mask'] = np.array(features['mask'])
                else:
                    features['embedding'] = np.array(features['embedding'])
                
                self.database[person_id][eye_label] = {
                    'features': features,
                    'quality': eye_data['quality'],
                    'metadata': eye_data['metadata']
                }

# Configuration file example (iris_config.json)
"""
{
  "use_deep_learning": false,
  "quality_threshold": 70,
  "verification_threshold": 0.68,
  "identification_threshold": 0.72,
  "segmentation": {
    "method": "daugman",
    "pupil_radius_range": [20, 80],
    "iris_radius_range": [80, 150]
  },
  "encoding": {
    "n_scales": 4,
    "n_orientations": 4,
    "angular_resolution": 256,
    "radial_resolution": 32
  },
  "matching": {
    "threshold": 0.32,
    "rotation_range": [-10, 10]
  },
  "models": {
    "segmentation": "models/iris_unet.pth",
    "features": "models/iris_features.pth"
  }
}
"""

# Usage example
if __name__ == "__main__":
    # Initialize system
    iris_system = IrisRecognitionSystem('iris_config.json')
    
    # Enroll iris
    result = iris_system.enroll_iris(
        'person1_left.png',
        'person_001',
        'left'
    )
    print(f"Enrollment: {result}")
    
    # Verify iris
    result = iris_system.verify_iris(
        'query_left.png',
        'person_001',
        'left'
    )
    print(f"Verification: {result}")
    
    # Identify iris
    result = iris_system.identify_iris('unknown_iris.png')
    print(f"Identification: {result}")
    
    # Save database
    iris_system.save_database('iris_database.json')
```

## Challenges & Solutions

### NIR vs Visible Light
```python
class VisibleLightIris:
    """Handle visible light iris images"""
    
    def __init__(self):
        self.color_channels = ['R', 'G', 'B']
    
    def preprocess_visible_iris(self, rgb_image):
        """Preprocess visible light iris image"""
        # Extract channels
        r, g, b = cv2.split(rgb_image)
        
        # Red channel often provides best contrast
        # But combine information from all channels
        
        # Method 1: Weighted combination
        weighted = 0.6 * r + 0.3 * g + 0.1 * b
        
        # Method 2: Red channel with blue suppression
        iris_enhanced = r - 0.5 * b
        iris_enhanced = np.clip(iris_enhanced, 0, 255).astype(np.uint8)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(iris_enhanced)
        
        # Reduce specular reflections
        enhanced = self.remove_specular_reflections(enhanced)
        
        return enhanced
    
    def remove_specular_reflections(self, image):
        """Remove specular reflections from visible light images"""
        # Detect bright spots
        _, bright_mask = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
        
        # Dilate to cover reflection areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.dilate(bright_mask, kernel)
        
        # Inpaint bright regions
        inpainted = cv2.inpaint(image, bright_mask, 3, cv2.INPAINT_TELEA)
        
        return inpainted
```

### Off-Angle Iris Recognition
```python
class OffAngleIrisRecognition:
    """Handle non-frontal iris images"""
    
    def __init__(self):
        self.max_angle = 30  # degrees
    
    def estimate_gaze_angle(self, image, segmentation):
        """Estimate gaze angle from iris ellipse"""
        iris_x, iris_y, iris_r = segmentation['iris']
        pupil_x, pupil_y, pupil_r = segmentation['pupil']
        
        # Fit ellipses to boundaries
        iris_points = self.extract_boundary_points(image, iris_x, iris_y, iris_r)
        
        if len(iris_points) > 5:
            iris_ellipse = cv2.fitEllipse(iris_points)
            (cx, cy), (major_axis, minor_axis), angle = iris_ellipse
            
            # Estimate off-angle from eccentricity
            eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2)
            
            # Approximate gaze angle
            gaze_angle = np.arcsin(eccentricity) * 180 / np.pi
            
            return {
                'angle': gaze_angle,
                'direction': angle,
                'confidence': 1.0 - eccentricity
            }
        
        return None
    
    def correct_perspective(self, image, gaze_info):
        """Correct perspective distortion"""
        if gaze_info['angle'] > self.max_angle:
            return None  # Too extreme
        
        # Compute homography for perspective correction
        h, w = image.shape
        
        # Source points (distorted)
        src_points = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        
        # Destination points (corrected)
        angle_rad = gaze_info['angle'] * np.pi / 180
        offset = int(w * np.tan(angle_rad) / 4)
        
        dst_points = np.float32([
            [offset, 0], [w-offset, 0], [w, h], [0, h]
        ])
        
        # Compute homography
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp image
        corrected = cv2.warpPerspective(image, H, (w, h))
        
        return corrected
```

## Security & Anti-Spoofing

### Liveness Detection
```python
class IrisLivenessDetection:
    """Detect presentation attacks on iris"""
    
    def __init__(self):
        self.methods = {
            'pupil_dynamics': self.check_pupil_dynamics,
            'texture_analysis': self.analyze_texture_liveness,
            'specular_reflection': self.check_specular_pattern,
            'frequency_analysis': self.analyze_frequency_spectrum
        }
    
    def detect_liveness(self, image_sequence):
        """Comprehensive liveness detection"""
        results = {}
        
        # Single frame methods
        if len(image_sequence) == 1:
            image = image_sequence[0]
            results['texture'] = self.analyze_texture_liveness(image)
            results['specular'] = self.check_specular_pattern(image)
            results['frequency'] = self.analyze_frequency_spectrum(image)
        else:
            # Multi-frame methods
            results['pupil_dynamics'] = self.check_pupil_dynamics(image_sequence)
            
            # Also apply single-frame methods to best frame
            best_frame = self.select_best_frame(image_sequence)
            results['texture'] = self.analyze_texture_liveness(best_frame)
            results['specular'] = self.check_specular_pattern(best_frame)
        
        # Combine scores
        liveness_score = self.combine_liveness_scores(results)
        
        return {
            'is_live': liveness_score > 0.5,
            'score': liveness_score,
            'details': results
        }
    
    def check_pupil_dynamics(self, image_sequence):
        """Check pupil hippus (natural oscillations)"""
        pupil_sizes = []
        
        for image in image_sequence:
            segmentation = self.quick_segment(image)
            if segmentation:
                pupil_sizes.append(segmentation['pupil'][2])  # radius
        
        if len(pupil_sizes) < 5:
            return 0.5  # Inconclusive
        
        # Analyze pupil size variations
        pupil_sizes = np.array(pupil_sizes)
        
        # Natural pupil shows small oscillations (hippus)
        variations = np.diff(pupil_sizes)
        
        # Features
        std_variation = np.std(variations)
        mean_size = np.mean(pupil_sizes)
        
        # Hippus typically 0.5-3% of pupil diameter
        relative_variation = std_variation / mean_size
        
        if 0.005 < relative_variation < 0.03:
            return 0.9  # Likely live
        elif relative_variation < 0.001:
            return 0.1  # Too static, likely fake
        else:
            return 0.5  # Uncertain
    
    def analyze_texture_liveness(self, image):
        """Analyze iris texture for liveness"""
        # Segment iris
        segmentation = self.quick_segment(image)
        if not segmentation:
            return 0.5
        
        # Extract iris region
        iris_mask = self.create_iris_mask(image.shape, segmentation)
        iris_pixels = image[iris_mask]
        
        # Texture features
        features = {}
        
        # 1. Local Binary Patterns
        lbp = self.compute_lbp(image, iris_mask)
        features['lbp_energy'] = np.sum(lbp**2) / len(lbp)
        
        # 2. Gray Level Co-occurrence Matrix
        glcm = self.compute_glcm(iris_pixels)
        features['glcm_contrast'] = self.glcm_contrast(glcm)
        features['glcm_homogeneity'] = self.glcm_homogeneity(glcm)
        
        # 3. Wavelet features
        wavelet_features = self.compute_wavelet_features(iris_pixels)
        features.update(wavelet_features)
        
        # Classification (simple threshold-based)
        # Real irises have higher texture complexity
        texture_score = (
            0.3 * (features['lbp_energy'] > 1000) +
            0.3 * (features['glcm_contrast'] > 50) +
            0.4 * (features['wavelet_energy'] > 100)
        )
        
        return texture_score
    
    def check_specular_pattern(self, image):
        """Check specular reflection patterns"""
        # Detect bright spots
        bright_threshold = np.percentile(image, 99)
        bright_mask = image > bright_threshold
        
        # Analyze bright regions
        num_labels, labels = cv2.connectedComponents(bright_mask.astype(np.uint8))
        
        # Real eyes have 1-2 corneal reflections
        # Printed/display attacks may have different patterns
        
        if 1 <= num_labels - 1 <= 3:  # Subtract 1 for background
            # Check shape and position of reflections
            reflection_score = 0.8
            
            for label in range(1, num_labels):
                region = (labels == label)
                
                # Check circularity
                contours, _ = cv2.findContours(
                    region.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    area = cv2.contourArea(contours[0])
                    perimeter = cv2.arcLength(contours[0], True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        if circularity < 0.5:  # Not circular enough
                            reflection_score *= 0.7
            
            return reflection_score
        else:
            return 0.3  # Suspicious pattern
    
    def analyze_frequency_spectrum(self, image):
        """Analyze frequency spectrum for print artifacts"""
        # FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Look for printing artifacts (regular patterns)
        # Real irises have more natural frequency distribution
        
        # Radial average
        center = np.array(magnitude.shape) // 2
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        radial_profile = np.bincount(r.ravel(), magnitude.ravel()) / np.bincount(r.ravel())
        
        # Check for peaks indicating regular patterns
        peaks = self.find_peaks(radial_profile[10:50])  # Mid frequencies
        
        if len(peaks) > 3:
            return 0.2  # Many peaks suggest printing artifacts
        else:
            return 0.8  # Natural spectrum
    
    def combine_liveness_scores(self, scores):
        """Combine individual liveness scores"""
        # Weighted combination
        weights = {
            'pupil_dynamics': 0.4,
            'texture': 0.3,
            'specular': 0.2,
            'frequency': 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for method, score in scores.items():
            if method in weights:
                total_score += weights[method] * score
                total_weight += weights[method]
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5  # Default uncertain
```

### Template Protection
```python
class IrisTemplateProtection:
    """Cancelable biometrics for iris templates"""
    
    def __init__(self):
        self.transform_key = None
    
    def generate_cancelable_template(self, iris_code, user_key):
        """Generate cancelable iris template"""
        # Random projection
        n_bits = len(iris_code)
        n_projections = n_bits // 2
        
        # Generate user-specific random matrix
        np.random.seed(hash(user_key) % (2**32))
        projection_matrix = np.random.randn(n_projections, n_bits)
        
        # Binary random projection
        projected = np.dot(projection_matrix, iris_code)
        cancelable_code = (projected > 0).astype(np.uint8)
        
        # Store transformation
        self.transform_key = {
            'matrix': projection_matrix,
            'original_size': n_bits
        }
        
        return cancelable_code
    
    def revoke_and_reissue(self, iris_code, new_key):
        """Revoke old template and issue new one"""
        # Generate completely new transformation
        return self.generate_cancelable_template(iris_code, new_key)
    
    def match_cancelable_templates(self, template1, template2):
        """Match cancelable templates"""
        # Standard Hamming distance on transformed space
        distance = np.sum(template1 != template2) / len(template1)
        return 1.0 - distance
```

## Resources

### Research Papers
- **[How Iris Recognition Works](https://ieeexplore.ieee.org/document/1262028)** - Daugman, 2004
- **[IrisCode](https://www.cl.cam.ac.uk/~jgd1000/irisrecog.pdf)** - Original Daugman paper
- **[Deep Learning for Iris Recognition](https://arxiv.org/abs/1907.09380)** - Survey, 2019
- **[Iris Recognition: A Review](https://ieeexplore.ieee.org/document/8580813)** - 2019

### Books
- "Handbook of Iris Recognition" - Bowyer, Hollingsworth, Flynn
- "Iris Recognition: An Emerging Biometric Technology" - IEEE
- "Biometric Systems: Technology, Design and Performance Evaluation" - Wayman

### Tools & Software
- **[OSIRIS](http://svnext.it-sudparis.eu/svnview2-eph/ref_syst/Iris_Osiris_v4.1/)** - Open Source Iris Recognition
- **[VASIR](https://www.nist.gov/services-resources/software/vasir-video-based-automated-system-iris-recognition)** - NIST's iris software
- **[LibIris](https://github.com/CVRL/libIris)** - Notre Dame iris library

### Online Resources
- **[ISO/IEC Standards](https://www.iso.org/committee/313770.html)** - Biometric standards
- **[NIST IREX](https://www.nist.gov/programs-projects/iris-exchange-irex-overview)** - Iris evaluation
- **[IEEE Biometrics Council](http://ieee-biometrics.org/)** - Research community
- **[Notre Dame CVRL](https://cvrl.nd.edu/)** - Computer Vision Research Lab