# CoMet: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning

![](imgs/cover.png)

**CoMet: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning**  
*Muhammad Aqeel, Shakiba Sharifi, Marco Cristani, Francesco Setti*  
**ICCV 2025** | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Aqeel_Towards_Real_Unsupervised_Anomaly_Detection_Via_Confident_Meta-Learning_ICCV_2025_paper.html) | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Aqeel_Towards_Real_Unsupervised_Anomaly_Detection_Via_Confident_Meta-Learning_ICCV_2025_paper.pdf)

---

## Introduction

**CoMet** is a **confidence-aware meta-learning framework** for **unsupervised image anomaly detection and localization**. It leverages:

- **Meta-learning** for rapid adaptation to unseen anomaly types  
- **Confidence modeling** to suppress false positives  
- **Synthetic anomaly augmentation** via lightweight GANs  
- **No external data** — fully unsupervised

Achieves **state-of-the-art performance** on **MVTec AD**, **VIADUCT**, and **KSDD2** with **minimal complexity**.

---

## Key Features

- **Simple & clean** PyTorch implementation  
- **No pre-trained detectors** (unlike PatchCore, PaDiM)  
- **Fast inference** (~30 FPS on RTX 4090)  
- **Visualizes anomaly heatmaps** with score overlay  
- **Supports 30+ backbones** via `timm` and `torchvision`

---

## Get Started

### Environment

```bash
pip install torch==1.12.1 torchvision==0.13.1 timm scikit-learn pandas opencv-python scikit-image