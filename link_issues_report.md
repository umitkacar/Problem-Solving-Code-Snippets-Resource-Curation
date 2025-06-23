# Link Issues Report - Awesome AI Resources

Generated: 2025-06-20

## Summary

Total URLs scanned: 721 across 65 markdown files
Total issues identified: 100+

## Critical Issues

### 1. Missing LICENSE File
- **README.md:9** - References `LICENSE` file that doesn't exist in repository
  - This should either be created or the link should be removed

### 2. HTTP URLs That Should Be HTTPS (54 found)

#### High Priority (Academic/Official Sites):
- **ReinforcementLearning/rl-basics.md:211** - `http://incompleteideas.net/book/the-book-2nd.html`
- **AutoML/automl-frameworks.md:52** - `http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html`
- **AutoML/automl-frameworks.md:125** - `http://epistasislab.github.io/tpot/`
- **AutoML/automl-frameworks.md:260** - `http://hyperopt.github.io/hyperopt/`
- **Biometrics/README.md:112** - `http://dlib.net/`
- **Biometrics/README.md:113** - `http://openbiometrics.org/`
- **Biometrics/README.md:165** - `http://vis-www.cs.umass.edu/lfw/`
- **Biometrics/README.md:170** - `http://bias.csr.unibo.it/fvc2006/`
- **Biometrics/README.md:174** - `http://biometrics.idealtest.org/`
- **Audio/speech-recognition.md:103** - `http://www.openslr.org/12/` (LibriSpeech dataset)

### 3. Potentially Broken or Irrelevant Links

#### Mismatched Content:
- **Biometrics/face-recognition.md:313** - Text: "CASIA-WebFace" links to `https://github.com/happynear/AMSoftmax` (should link to dataset, not implementation)
- **GraphNeuralNetworks/gnn-fundamentals.md:297** - Text: "Facebook" links to Stanford SNAP data (confusing)
- **Biometrics/voice-recognition.md:1418** - Text: "LibriSpeech" links to `https://www.openslr.org/12/` (HTTP issue + unclear if correct)

#### Version-Specific Documentation (May Be Outdated):
- **ReinforcementLearning/rl-basics.md:108** - `https://docs.ray.io/en/latest/rllib/index.html`
- **AutoML/automl-frameworks.md:253** - `https://docs.ray.io/en/latest/tune/index.html`

### 4. Placeholder/Example URLs
- **Mobile/ai-flutter.md:242** - `https://api.your-nlp-service.com/sentiment'`
- **Tools/mlops.md:347** - `https://github.com/your-org/ml-notebooks.git"`

### 5. Local File References (12 found)
These are actually fine for internal navigation:
- **README.md:39-41** - Links to `./notebooks/` subdirectories
- **notebooks/README.md** - Multiple links to `.ipynb` files

### 6. Malformed URLs (5 found)
These appear to be code snippets incorrectly parsed as URLs:
- **ComputerVision/video-inpainting.md:632** - `decoded[:, t]`
- **GenerativeAI/image-enhancement.md:1419** - `img_tensor, scale_factor`
- **GenerativeAI/image-enhancement.md:1421** - `img_tensor, denoise_strength`
- **Biometrics/voice-recognition.md:145** - `audio, sr`

### 7. Potential 404s or Changed URLs

Based on common patterns, these URLs may be broken:
- Microsoft Research links (often change/remove projects)
- Dataset links on university servers
- GitHub blob links to images (should use raw.githubusercontent.com)

## Recommendations

1. **Immediate Actions:**
   - Create LICENSE file or remove the link
   - Update all HTTP URLs to HTTPS where applicable
   - Fix placeholder URLs in ai-flutter.md and mlops.md

2. **Manual Verification Needed:**
   - All Microsoft Research project links
   - Academic dataset links (especially biometrics datasets)
   - Version-specific documentation links

3. **Content Fixes:**
   - Fix mismatched link descriptions (e.g., CASIA-WebFace)
   - Remove code snippets that were incorrectly identified as URLs

4. **Future Prevention:**
   - Add automated link checking to CI/CD
   - Prefer persistent URLs (DOI, official repos) over temporary ones
   - Document URL update process in CONTRIBUTING.md