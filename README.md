# ViT-LLM for Automated Medical Image Analysis

A Vision-Language Framework for Chest X-ray Report Generation

## 🏥 Overview

This project presents an innovative ViT-LLM framework that combines Vision Transformers (ViT) with Large Language Models (LLMs) to automatically generate diagnostic reports from chest X-ray images. The system addresses the critical shortage of radiologists, particularly in rural areas, by providing automated, reliable X-ray interpretation solutions.

## 🎯 Motivation

- **Critical Healthcare Gap**: 65% of specialist doctor positions in Indian government health facilities remain unfilled
- **Rural Healthcare Crisis**: 70% shortfall in specialist doctors at rural Community Health Centres (CHCs)
- **Growing Demand**: Exponential growth in imaging data causing diagnostic backlogs
- **Manual Limitations**: Time-consuming report generation prone to inter-observer variability

## 🔬 Key Features

- **Multi-Modal Architecture**: Integration of Vision Transformers for image analysis and LLMs for report generation
- **Multi-View Processing**: Supports multiple X-ray views per report (up to 5 images)
- **Domain Adaptation**: Specialized fine-tuning on medical datasets
- **Self-Attention Fusion**: Captures inter-image dependencies and relationships
- **Progressive Learning**: Three distinct techniques with increasing complexity

## 🏗️ Architecture

### Three Progressive Techniques:

1. **Technique 1**: Vision Transformer and LLM trained from scratch
   - Single frontal-view chest X-ray per report
   - End-to-end training from scratch

2. **Technique 2**: ViT + Pretrained BART + Multi-Image Fusion
   - Multiple X-ray views per report (up to 5)
   - Self-attention fusion module
   - Pretrained BART decoder

3. **Technique 3**: Fine-Tuned ViT and BART with Domain Adaptation ⭐
   - ViT fine-tuned on RSNA Pneumonia Dataset
   - BART fine-tuned on radiology-specific text
   - Multi-image input with self-attention fusion

## 📊 Performance Results

Our best model (Technique 3) achieved state-of-the-art performance:

| Model | METEOR | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 |
|-------|--------|---------|---------|---------|---------|
| **Proposed Technique 3** | **0.2605** | **0.3890** | **0.3565** | **0.2991** | **0.0874** |
| Proposed Technique 2 | 0.218 | 0.346 | 0.324 | 0.269 | 0.078 |
| Proposed Technique 1 | 0.182 | 0.317 | 0.294 | 0.239 | 0.056 |
| CDGPT2 | 0.164 | 0.289 | — | 0.387 | 0.111 |
| TrMRG | 0.218 | 0.387 | — | 0.532 | 0.158 |

## 📚 Dataset

### Training Data Sources:
- **RSNA Pneumonia Processed Dataset**: For ViT encoder fine-tuning
- **Radiology-Specific Text Corpus**: From open-access clinical repositories
- **Kaggle Radiology Datasets**: Publicly available chest X-ray reports

### Preprocessing Pipeline:
- Image resizing to standard dimensions
- Contrast normalization
- Data augmentation techniques
- Multiple view alignment for multi-image models

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/vit-llm-medical-analysis.git
cd vit-llm-medical-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from vit_llm_medical import ViTLLMModel

# Initialize the model
model = ViTLLMModel(
    technique=3,  # Use best performing technique
    model_path="path/to/pretrained/model"
)

# Generate report from chest X-ray(s)
images = ["path/to/xray1.jpg", "path/to/xray2.jpg"]
report = model.generate_report(images)
print(report)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- OpenCV 4.5+
- NumPy
- Pillow
- scikit-learn
- matplotlib

## 🎯 Usage Examples

### Single Image Analysis
```python
# Analyze single chest X-ray
model = ViTLLMModel(technique=1)
report = model.generate_report("single_xray.jpg")
```

### Multi-Image Analysis
```python
# Analyze multiple views
model = ViTLLMModel(technique=3)
images = ["frontal_view.jpg", "lateral_view.jpg", "pa_view.jpg"]
comprehensive_report = model.generate_report(images)
```

### Custom Fine-tuning
```python
# Fine-tune on custom dataset
model = ViTLLMModel(technique=3)
model.fine_tune(
    image_dir="path/to/images",
    report_dir="path/to/reports",
    epochs=50,
    learning_rate=1e-4
)
```

## 📈 Evaluation Metrics

The model is evaluated using standard NLG metrics:

- **METEOR**: Semantic alignment with reference reports
- **ROUGE-1/ROUGE-L**: N-gram overlap and longest common subsequence
- **BLEU-1/BLEU-4**: Precision of unigram and 4-gram matches

## 🔮 Future Work

- **Cross-Modality Integration**: Extend to CT scans, MRI, and other radiography
- **Clinical Metadata Integration**: Incorporate patient history and demographics
- **Expert-in-the-Loop Feedback**: Continuous learning from radiologist feedback
- **Active Learning Implementation**: Improve model performance through selective data annotation

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{singh2025vit,
  title={ViT-LLM for Automated Medical Image Analysis: A Vision-Language Framework for Chest X-ray Report Generation},
  author={Singh, Ayush Kumar and Patra, Rusha},
  institution={Indian Institute of Information Technology, Guwahati},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ayush Kumar Singh** - *Lead Developer* - [GitHub Profile](https://github.com/ayushkumarsingh)
- **Dr. Rusha Patra** - *Supervisor* - Indian Institute of Information Technology, Guwahati

## 🙏 Acknowledgments

- Indian Institute of Information Technology, Guwahati
- RSNA for the Pneumonia Dataset
- Open-source contributors to PyTorch and Transformers libraries
- Medical professionals who provided expert annotations

## 📞 Contact

For questions and support:
- Email: ayush.kumar.singh@iiitg.ac.in
- Institution: Indian Institute of Information Technology, Guwahati

---

**Disclaimer**: This tool is designed to assist medical professionals and should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.
