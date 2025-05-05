# DAYKD

**Software Code Availability Information**  

The source code and implementation details for the **Dual Attentive YOLOv11 Keypoint Detector (DAYKD)** model presented in **End-to-End Model Enabled GPR Hyperbolic Keypoint Detection for Automatic Localization of Underground Targets** are available to facilitate reproducibility and further research. Below is the relevant information:  

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

### **Support**  
For technical queries, contact the corresponding author:  
- **Email**: [houfeifei@csu.edu.cn](mailto:houfeifei@csu.edu.cn)  

### **Note**  
The simulated dataset was generated using **gprMax** (v3.1.5), and real-world data will be shared upon request due to privacy agreements.  

### **Acknowledgments**
This project uses code from [YOLOv11](https://github.com/ultralytics/ultralytics), which is licensed under [AGPL-3.0 License]. Thanks to the original authors for their work.

### Usage Instructions
- The pre-trained YOLOv11 model is located in the `runs/train-pose` folder.  
- Sample test data for inference is provided in the `image` folder.  
- Run `pose-predict.py` to perform predictions on the data.  

### Notes:  
- This model is an improved version of YOLOv11.  
- Ensure your environment is properly configured to **train and test** the original YOLOv11 model before running this code.  
