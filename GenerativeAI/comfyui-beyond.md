<div align="center">

<!-- Animated Typing SVG Header -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=2800&pause=2000&color=00D9FF&center=true&vCenter=true&width=940&lines=ComfyUI+%26+Beyond+%F0%9F%94%A5;Node-Based+Generative+AI+Workflows;Production-Ready+2024-2025" alt="Typing SVG" />

<!-- Modern Shields -->
<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Latest-00D9FF?style=for-the-badge&logo=node.js" alt="ComfyUI"/>
  <img src="https://img.shields.io/badge/SD_3.5-Supported-blueviolet?style=for-the-badge" alt="SD 3.5"/>
  <img src="https://img.shields.io/badge/Flux-Compatible-ec4899?style=for-the-badge" alt="Flux"/>
  <img src="https://img.shields.io/badge/SDXL-Optimized-orange?style=for-the-badge" alt="SDXL"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/1000+-Custom_Nodes-success?style=for-the-badge" alt="Custom Nodes"/>
  <img src="https://img.shields.io/badge/Production-Ready-green?style=for-the-badge" alt="Production Ready"/>
  <img src="https://img.shields.io/badge/2024--2025-Latest-red?style=for-the-badge&logo=calendar" alt="2024-2025"/>
</p>

---

### 🎯 Master Node-Based AI Workflows
*From Beginner to Advanced: Build Production-Ready Generative AI Pipelines*

[![GitHub Stars](https://img.shields.io/github/stars/comfyanonymous/ComfyUI?style=social)](https://github.com/comfyanonymous/ComfyUI)
[![Last Updated](https://img.shields.io/badge/Last_Updated-November_2025-success?style=flat-square)](https://github.com)

</div>

---

## 📋 Table of Contents

- [🚀 What's New in 2024-2025](#-whats-new-in-2024-2025)
- [⚡ Quick Start](#-quick-start)
- [🎨 Essential Workflows](#-essential-workflows)
- [🔌 Top Custom Nodes 2024-2025](#-top-custom-nodes-2024-2025)
- [🏗️ Advanced Architectures](#️-advanced-architectures)
- [📊 Workflow Examples](#-workflow-examples)
- [🎯 Production Deployment](#-production-deployment)
- [💡 Tips & Optimization](#-tips--optimization)
- [🌐 Community & Resources](#-community--resources)

---

## 🚀 What's New in 2024-2025

### Latest Features

```mermaid
timeline
    title ComfyUI Evolution 2023-2025
    2023-Q1 : Initial Release
            : Basic Workflows
    2023-Q4 : SDXL Support
            : ControlNet Integration
    2024-Q1 : Flux.1 Compatible
            : IP-Adapter Plus
    2024-Q2 : SD 3.5 Support
            : Video Generation
    2024-Q3 : LTX Video
            : Advanced Samplers
    2024-Q4 : Workspace Manager
            : API V2
    2025-Q1 : Real-time Preview
            : Cloud Integration
```

### 🏆 Key Advantages

| Feature | ComfyUI | A1111 WebUI | InvokeAI |
|---------|---------|-------------|----------|
| **Memory Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Workflow Complexity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Custom Nodes** | 1000+ | Extensions | Limited |
| **Learning Curve** | Moderate | Easy | Moderate |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Batch Processing** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **API Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For NVIDIA GPUs (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Launch ComfyUI
python main.py

# Access at http://127.0.0.1:8188
```

### Installation with Manager (Recommended)

```bash
# After basic install, add ComfyUI Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Restart ComfyUI - Manager will appear in the interface
```

### 📁 Directory Structure

```mermaid
graph TD
    A[ComfyUI/] --> B[models/]
    A --> C[custom_nodes/]
    A --> D[input/]
    A --> E[output/]

    B --> B1[checkpoints/]
    B --> B2[loras/]
    B --> B3[vae/]
    B --> B4[controlnet/]
    B --> B5[clip/]
    B --> B6[upscale_models/]

    C --> C1[ComfyUI-Manager/]
    C --> C2[efficiency-nodes/]
    C --> C3[IPAdapter-plus/]
    C --> C4[AnimateDiff-Evolved/]

    style A fill:#00D9FF,stroke:#0099CC,stroke-width:3px,color:#000
    style B fill:#a855f7,stroke:#7e22ce,stroke-width:2px,color:#fff
    style C fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
```

---

## 🎨 Essential Workflows

### Basic Text-to-Image Workflow

```mermaid
flowchart LR
    A[Checkpoint Loader] --> B[CLIP Text Encode<br/>Positive]
    A --> C[CLIP Text Encode<br/>Negative]
    D[Empty Latent Image] --> E[KSampler]
    B --> E
    C --> E
    A --> E
    E --> F[VAE Decode]
    A --> F
    F --> G[Save Image]

    style A fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    style E fill:#a855f7,stroke:#7e22ce,stroke-width:2px,color:#fff
    style G fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
```

### Advanced SDXL Workflow with Refiner

```mermaid
flowchart TD
    A[SDXL Base Model] --> B[Positive Conditioning]
    A --> C[Negative Conditioning]
    D[Empty Latent] --> E[KSampler Base<br/>Steps: 20]
    B --> E
    C --> E
    A --> E

    E --> F[SDXL Refiner Model]
    B2[Refiner Positive] --> G[KSampler Refiner<br/>Steps: 10]
    C2[Refiner Negative] --> G
    F --> G
    E --> G

    G --> H[VAE Decode]
    F --> H
    H --> I[Save Image]

    style E fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style G fill:#ec4899,stroke:#be185d,stroke-width:2px,color:#fff
```

---

## 🔌 Top Custom Nodes 2024-2025

### 🏆 Essential Nodes

#### 1. **ComfyUI Manager** ⭐⭐⭐⭐⭐
[![Stars](https://img.shields.io/github/stars/ltdrdata/ComfyUI-Manager?style=social)](https://github.com/ltdrdata/ComfyUI-Manager)

**Features:**
- Install/update custom nodes with one click
- Model manager
- Missing nodes auto-install
- Workflow sharing

```bash
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

#### 2. **Efficiency Nodes** ⭐⭐⭐⭐⭐
[![Stars](https://img.shields.io/github/stars/jags111/efficiency-nodes-comfyui?style=social)](https://github.com/jags111/efficiency-nodes-comfyui)

**Features:**
- Consolidated nodes for faster workflows
- XY Plot generation
- Highres-Fix node
- Script nodes for automation

#### 3. **IP-Adapter Plus** ⭐⭐⭐⭐⭐
[![Stars](https://img.shields.io/github/stars/cubiq/ComfyUI_IPAdapter_plus?style=social)](https://github.com/cubiq/ComfyUI_IPAdapter_plus)

**Use Cases:**
- Style transfer from reference images
- Face ID preservation
- Composition guidance
- Multi-image conditioning

**Workflow:**
```mermaid
flowchart LR
    A[Base Model] --> B[IP-Adapter Apply]
    C[Reference Image] --> D[IP-Adapter Encoder]
    D --> B
    E[Text Prompt] --> B
    B --> F[Generate]

    style B fill:#ec4899,stroke:#be185d,stroke-width:2px,color:#fff
```

#### 4. **ControlNet Preprocessors** ⭐⭐⭐⭐⭐
[![Stars](https://img.shields.io/github/stars/Fannovel16/comfyui_controlnet_aux?style=social)](https://github.com/Fannovel16/comfyui_controlnet_aux)

**Available Preprocessors:**
- ✅ Canny Edge Detection
- ✅ Depth (MiDaS, ZoeDepth, DepthAnything)
- ✅ Normal Map
- ✅ OpenPose & DWPose
- ✅ Lineart (Anime, Realistic)
- ✅ Scribble & HED
- ✅ Segmentation (OneFormer, SAM)

#### 5. **AnimateDiff Evolved** ⭐⭐⭐⭐⭐
[![Stars](https://img.shields.io/github/stars/Kosinkadink/ComfyUI-AnimateDiff-Evolved?style=social)](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)

**Video Generation:**
- AnimateDiff for SD 1.5 and SDXL
- Motion LoRAs
- HotshotXL support
- Frame interpolation
- Context scheduling

### 🌟 2024-2025 New Nodes

#### 6. **InstantID**
Face-preserving generation with incredible consistency

```python
# Node setup
instantid_model -> apply_instantid -> ksampler
face_image -> face_analysis -> apply_instantid
```

#### 7. **PhotoMaker**
Photorealistic portrait generation

#### 8. **LTX Video**
State-of-the-art video generation (2024)

#### 9. **IC-Light**
Controllable relighting in generation

#### 10. **LayerDiffuse**
Transparent image generation with alpha channel

---

## 🏗️ Advanced Architectures

### Multi-ControlNet + IP-Adapter Workflow

```mermaid
flowchart TB
    A[SDXL Model] --> M[Multi-Apply]

    subgraph Controls
        C1[Canny Image] --> CN1[ControlNet Canny]
        C2[Depth Image] --> CN2[ControlNet Depth]
        C3[Pose Image] --> CN3[ControlNet Pose]
    end

    subgraph Style
        S1[Style Reference] --> IP[IP-Adapter]
    end

    CN1 --> M
    CN2 --> M
    CN3 --> M
    IP --> M

    T[Text Conditioning] --> M
    M --> K[KSampler Advanced]
    K --> V[VAE Decode]
    V --> O[Output]

    style M fill:#a855f7,stroke:#7e22ce,stroke-width:3px,color:#fff
    style K fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style O fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
```

### Upscale Workflow (4K+)

```mermaid
flowchart LR
    A[Input Image] --> B[Upscale Model<br/>4x-UltraSharp]
    B --> C[Load Image]
    C --> D[VAE Encode]
    D --> E[KSampler<br/>img2img<br/>Denoise: 0.3]
    F[Base Model] --> E
    G[Positive Prompt] --> E
    E --> H[VAE Decode]
    H --> I[Save 4K Image]

    style B fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    style E fill:#a855f7,stroke:#7e22ce,stroke-width:2px,color:#fff
    style I fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
```

---

## 📊 Workflow Examples

### 1. Photorealistic Portrait Pipeline

**Nodes Used:**
- Checkpoint: `realisticVisionV60.safetensors`
- ControlNet: OpenPose + Depth
- IP-Adapter: Face ID
- Upscaler: 4x-UltraSharp
- Face Restore: CodeFormer

**Settings:**
```json
{
  "base_steps": 30,
  "cfg": 7.0,
  "sampler": "dpmpp_2m_sde_gpu",
  "scheduler": "karras",
  "denoise": 1.0,
  "controlnet_strength": [0.7, 0.5],
  "ip_adapter_weight": 0.6
}
```

### 2. Architectural Visualization

**Workflow:**
```mermaid
graph LR
    A[Sketch Input] --> B[Canny Preprocessor]
    B --> C[ControlNet Canny]
    D[SDXL Architecture Fine-tune] --> E[KSampler]
    C --> E
    F[Professional Photography Prompt] --> E
    E --> G[Refiner]
    G --> H[Upscale 2x]
    H --> I[Final Render]
```

### 3. Anime Character Generation

**Stack:**
- Model: `animagineXL3.safetensors`
- LoRAs: Character style + Pose control
- ControlNet: Lineart
- Additional: Color palette guidance

### 4. Product Photography

**Complete Pipeline:**
```mermaid
flowchart TD
    A[Product Photo] --> B[Background Removal]
    B --> C[Inpainting]
    D[Studio Background Prompt] --> C
    C --> E[Lighting Enhancement<br/>IC-Light]
    E --> F[Color Correction]
    F --> G[Upscale]
    G --> H[Professional Result]

    style E fill:#fbbf24,stroke:#d97706,stroke-width:2px,color:#000
    style H fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
```

### 5. Video Generation (AnimateDiff)

**16-Frame Animation:**
```json
{
  "workflow": {
    "checkpoint": "sd15_base",
    "motion_module": "mm_sd_v15_v2",
    "motion_lora": "v2_lora_RollingAnticlockwise",
    "frames": 16,
    "fps": 8,
    "context_length": 16,
    "prompt_travel": {
      "0": "day scene, sunny",
      "8": "sunset scene, golden hour",
      "16": "night scene, stars"
    }
  }
}
```

---

## 🎯 Production Deployment

### API Usage

```python
import requests
import json
import base64
from io import BytesIO
from PIL import Image

class ComfyUIAPI:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        """Queue a workflow for execution"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=data
        )
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        """Get generated image"""
        data = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        url_values = urllib.parse.urlencode(data)

        with urllib.request.urlopen(
            f"http://{self.server_address}/view?{url_values}"
        ) as response:
            return response.read()

    def generate_image(self, prompt_text, negative_prompt=""):
        """Complete generation pipeline"""
        # Load workflow template
        workflow = self.load_workflow_template()

        # Update prompts
        workflow["6"]["inputs"]["text"] = prompt_text
        workflow["7"]["inputs"]["text"] = negative_prompt

        # Queue and wait
        response = self.queue_prompt(workflow)
        prompt_id = response['prompt_id']

        # Wait for completion and get image
        output_images = self.wait_for_completion(prompt_id)

        return output_images

# Usage
api = ComfyUIAPI()
images = api.generate_image(
    "a beautiful landscape, mountains, lake, sunset",
    "blurry, low quality"
)
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
WORKDIR /app
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /app/ComfyUI

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install essential custom nodes
WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/jags111/efficiency-nodes-comfyui.git && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# Expose port
EXPOSE 8188

# Run ComfyUI
WORKDIR /app/ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
```

**Build and Run:**
```bash
docker build -t comfyui:latest .
docker run --gpus all -p 8188:8188 -v $(pwd)/models:/app/ComfyUI/models comfyui:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: comfyui
spec:
  replicas: 2
  selector:
    matchLabels:
      app: comfyui
  template:
    metadata:
      labels:
        app: comfyui
    spec:
      containers:
      - name: comfyui
        image: comfyui:latest
        ports:
        - containerPort: 8188
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        volumeMounts:
        - name: models
          mountPath: /app/ComfyUI/models
        - name: output
          mountPath: /app/ComfyUI/output
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: comfyui-models
      - name: output
        persistentVolumeClaim:
          claimName: comfyui-output
---
apiVersion: v1
kind: Service
metadata:
  name: comfyui-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8188
  selector:
    app: comfyui
```

---

## 💡 Tips & Optimization

### Performance Optimization

```mermaid
graph TD
    A[Performance Optimization] --> B[VRAM Management]
    A --> C[Speed Improvements]
    A --> D[Quality Settings]

    B --> B1[VAE Tiling: ON]
    B --> B2[Model Offloading]
    B --> B3[Lowvram Mode]

    C --> C1[TensorRT]
    C --> C2[xFormers]
    C --> C3[Batch Processing]

    D --> D1[Sampler Selection]
    D --> D2[Step Optimization]
    D --> D3[CFG Balance]

    style A fill:#00D9FF,stroke:#0099CC,stroke-width:3px,color:#000
```

### Best Practices

**1. VRAM Optimization**
```python
# Low VRAM settings (8GB)
--lowvram  # Loads models on demand
--preview-method auto  # Efficient previews

# Medium VRAM (12-16GB)
--normalvram  # Standard mode

# High VRAM (24GB+)
# No flags needed, full performance
```

**2. Sampler Recommendations**

| Use Case | Best Sampler | Steps | CFG |
|----------|-------------|-------|-----|
| **Quality (slow)** | DPM++ 2M Karras | 25-30 | 7.0 |
| **Balanced** | DPM++ SDE Karras | 20-25 | 6.5 |
| **Speed** | LCM | 4-8 | 1.5 |
| **Photorealistic** | DPM++ 2M SDE GPU | 30-40 | 7.5 |
| **Anime** | Euler a | 20-28 | 7.0 |

**3. Workflow Optimization**
- Group related nodes
- Use reroute nodes for clean connections
- Save frequently used node groups
- Use workflow templates
- Enable auto-queue for batch processing

**4. Model Management**
```bash
# Organize models
models/
├── checkpoints/
│   ├── realistic/
│   ├── anime/
│   └── artistic/
├── loras/
│   ├── characters/
│   ├── styles/
│   └── concepts/
└── controlnet/
    ├── sd15/
    └── sdxl/
```

### Troubleshooting

**Common Issues:**

| Problem | Solution |
|---------|----------|
| Out of Memory | Enable `--lowvram` or reduce batch size |
| Slow Generation | Use faster samplers (LCM, DPM++ 2M) |
| Poor Quality | Increase steps, adjust CFG |
| Missing Nodes | Install via ComfyUI Manager |
| Black Images | Check VAE, try different one |
| Workflow Won't Load | Update custom nodes |

---

## 🌐 Community & Resources

### Official Links

[![ComfyUI GitHub](https://img.shields.io/badge/GitHub-ComfyUI-181717?style=for-the-badge&logo=github)](https://github.com/comfyanonymous/ComfyUI)
[![Documentation](https://img.shields.io/badge/Docs-Official-blue?style=for-the-badge)](https://comfyanonymous.github.io/ComfyUI_examples/)
[![Examples](https://img.shields.io/badge/Examples-Gallery-green?style=for-the-badge)](https://comfyanonymous.github.io/ComfyUI_examples/)

### Top Resources

**Workflow Sharing:**
- 🌐 [OpenArt](https://openart.ai/workflows) - Thousands of workflows
- 🎨 [CivitAI](https://civitai.com/models?type=Workflow) - Community workflows
- 📊 [ComfyWorkflows](https://comfyworkflows.com/) - Curated collection

**Learning:**
- 📺 [Olivio Sarikas](https://www.youtube.com/@OlivioSarikas) - Comprehensive tutorials
- 🎓 [Scott Detweiler](https://www.youtube.com/@scottdetweiler) - Advanced techniques
- 🔥 [Aitrepreneur](https://www.youtube.com/@Aitrepreneur) - Business applications

**Custom Nodes:**
- 📦 [ComfyUI Nodes Registry](https://ltdrdata.github.io/ComfyUI-Manager/)
- 🔌 [Custom Nodes List](https://github.com/ltdrdata/ComfyUI-Manager)

### Community

- **Discord:** [ComfyUI Official](https://discord.gg/comfyui)
- **Reddit:** [r/comfyui](https://reddit.com/r/comfyui)
- **GitHub:** [Discussions](https://github.com/comfyanonymous/ComfyUI/discussions)

### Recommended Extensions

```mermaid
mindmap
  root((ComfyUI Extensions))
    Essential
      Manager
      Efficiency Nodes
      WAS Node Suite
    Image Control
      ControlNet Aux
      IP-Adapter Plus
      InstantID
    Video
      AnimateDiff Evolved
      Frame Interpolation
      Video Helper Suite
    Utilities
      Image Resize
      Checkpoint Merger
      Prompt Stylers
    Advanced
      Custom Scripts
      Impact Pack
      Inspire Pack
```

---

<div align="center">

## 🎓 Learning Path

**Beginner** → Install + Basic workflow → Text-to-Image mastery
↓
**Intermediate** → ControlNet + LoRAs → Complex workflows
↓
**Advanced** → Custom nodes + API + Production deployment
↓
**Expert** → Workflow optimization + Custom integrations + Business solutions

---

## 🌟 Contributing

Share your workflows and help the community grow!

[![Contribute](https://img.shields.io/badge/Contribute-Workflows-success?style=for-the-badge)](CONTRIBUTING.md)
[![Share](https://img.shields.io/badge/Share-Knowledge-blue?style=for-the-badge)](https://github.com)

---

**Last Updated:** November 2025 | **Next Update:** Weekly

*Join 100K+ users creating amazing AI art with ComfyUI!*

</div>
