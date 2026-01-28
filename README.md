# Fire Detection System using MobileNetV2

A deep learning computer vision system for detecting fire vs non-fire images using transfer learning with PyTorch. This project implements a fine-tuned MobileNetV2 model optimized for accurate fire detection with minimal false positives.

## 🔥 Project Highlights
- **Transfer Learning**: Fine-tuned MobileNetV2 pre-trained on ImageNet
- **Optimized for Real Use**: 0.4 decision threshold reduces false alarms
- **Data Augmentation**: Random horizontal flips for better generalization
- **Performance Metrics**: Comprehensive classification report with precision/recall

## 🛠️ Technical Implementation
- **Framework**: PyTorch
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Fine-tuning Strategy**: Frozen early layers, train last 4 feature layers
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam (learning rate: 1e-4)
- **Image Processing**: 224x224 resolution, ImageNet normalization
- **Validation**: 80/20 train-validation split

## 📊 Model Architecture
MobileNetV2 (pretrained)
├── Features layers (frozen except last 4)
└── Custom classifier
└── Linear layer (1280 → 1 output)

## 📈 Results
- **Validation Accuracy**: 98.0% (final epoch)
- **Decision Threshold**: 0.4 (optimized for fire detection sensitivity)
- **Training**: 10 epochs with consistent improvement
- **Model Performance**: Achieved near-perfect classification with minimal false positives

**Sample Classification Report:**
text
          precision    recall  f1-score   support
    Fire       0.98      0.97      0.98       150
Non-Fire       0.97      0.98      0.98       150
accuracy                           0.98       300
macro avg 0.98 0.98 0.98 300
weighted avg 0.98 0.98 0.98 300

text

**Key Achievement:** 
- Reached **98% validation accuracy** demonstrating highly reliable fire detection
- Optimized threshold (0.4) prioritizes fire detection sensitivity while maintaining precision
- Transfer learning with MobileNetV2 enabled fast convergence to high accuracy

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/muhammadattaurrehman57-hash/fire-detection-cnn.git
cd fire-detection-cnn

# Install dependencies
pip install -r requirements.txt

# Update dataset path in fire_detection.py
# python fire_detection.py
📁 Project Structure
text
fire-detection-cnn/
├── fire_detection.py          # Main training script
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
└── dataset/                   # Fire vs Non-Fire images
    ├── fire/
    └── non_fire/
📝 Dataset Preparation
Create folder structure:

text
fire_dataset/
├── fire/          # Fire images
└── non_fire/      # Non-fire images
Place your images in respective folders

Update DATA_DIR path in the script

🧠 Development Notes
This project was developed as a learning exercise in transfer learning and computer vision. The implementation focuses on practical application of deep learning concepts to solve real-world safety challenges through accurate fire detection.

👨‍💻 Author
Atta Ur Rehman
AI/ML Student | Arfa Karim Technology Incubator
Specializing in Computer Vision & Deep Learning
GitHub Profile

📄 License
MIT License
