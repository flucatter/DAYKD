# DAYKD

**Software Code Availability Information**  

The source code and implementation details for the **Dual Attentive YOLOv11 Keypoint Detector (DAYKD)** model presented in this paper are available to facilitate reproducibility and further research. Below is the relevant information:  

### **Code Repository**  
- **Repository URL**: [(https://github.com/flucatter/DAYKD)]()  
- **License**: MIT License (Open-source, free for academic and non-commercial use).  
- **Programming Language**: Python (PyTorch framework).  

### **Dependencies**  
- **Python**: 3.8+  
- **PyTorch**: 1.12.0+  
- **CUDA**: 11.3 (for GPU acceleration)  
- **Additional Libraries**: OpenCV, NumPy, SciPy, Ultralytics YOLOv11, gprMax (for simulated dataset generation).  

### **Functionality**  
The provided code includes:  
1. **DAYKD Model Architecture**: Implementation of the Dual-Task YOLOv11 framework with CAFM and FRFN modules.  
2. **Training Scripts**: Configuration files for hyperparameters (learning rate, batch size, loss weights).  
3. **Dataset Preprocessing**: Tools for GPR image augmentation, cropping, and keypoint annotation.  
4. **Evaluation Metrics**: Scripts to compute *Precision*, *Recall*, *mAP50*, *F1-score*, and inference speed (FPS).  

### **Hardware Requirements**  
- **Training**: NVIDIA GPU (RTX 3050 or higher with 8GB VRAM recommended).  
- **Inference**: Compatible with CPU (slower) or edge devices (Jetson Xavier).  

### **Usage Instructions**  
1. Clone the repository and install dependencies via `pip install -r requirements.txt`.  
2. Prepare datasets (simulated or real-world GPR B-scans) in YOLO annotation format.  
3. Run training:  
   ```bash  
   python train.py --task detection  # Task 1: Target detection  
   python train.py --task keypoint   # Task 2: Keypoint refinement  
   ```  
4. For inference on new GPR scans:  
   ```bash  
   python detect.py --source path/to/image --weights daykd.pt  
   ```  

### **Support**  
For technical queries, contact the corresponding author:  
- **Email**: [houfeifei@csu.edu.cn](mailto:houfeifei@csu.edu.cn)  

### **Note**  
The simulated dataset was generated using **gprMax** (v3.1.5), and real-world data will be shared upon request due to privacy agreements.  

---  
This statement ensures compliance with journal guidelines (20000-character limit) while providing essential details for replication. Let me know if you need modifications!
