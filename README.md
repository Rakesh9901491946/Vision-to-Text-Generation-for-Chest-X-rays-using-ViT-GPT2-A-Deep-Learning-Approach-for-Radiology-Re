# 🩺 Radiology Report Generation from Chest X-rays using ViT-GPT2

This project demonstrates an end-to-end deep learning pipeline for automatically generating radiology reports from chest X-ray images using a Vision-Language Transformer architecture (`ViT + GPT-2`). The model is trained on the **Indiana University Chest X-ray dataset** and leverages `Hugging Face Transformers` for the vision encoder-decoder setup.

---

## 🧠 Model Architecture

- **Encoder**: Vision Transformer (ViT) - `google/vit-base-patch16-224-in21k`
- **Decoder**: GPT-2 language model - `gpt2`
- **Frameworks**: PyTorch, Hugging Face Transformers, TorchVision
- **Approach**: VisionEncoderDecoderModel

---

## 📁 Dataset

- **Source**: [Indiana University Chest X-ray Collection](https://openi.nlm.nih.gov/faq#collection)
- **Format**:
  - Chest X-ray images (JPEG/PNG)
  - Radiology reports (`impression` section used as ground truth)
  - Accompanied with projection metadata

---

## 🚀 Features

- Image preprocessing using `ViTImageProcessor`
- Text tokenization with GPT2 tokenizer
- End-to-end training loop with:
  - Custom `ChestXRayDataset` class
  - Cross-entropy loss with padding ignored
  - Mixed precision support
- Inference for new images
- Report generation with beam search
- Saves model/tokenizer/processor for deployment

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/radiology-report-generation.git
cd radiology-report-generation

# Install requirements
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py
```

> Training uses the full dataset without explicit validation split. For real-world training, consider splitting into `train/val/test` sets using `sklearn.model_selection.train_test_split`.

---

## 🖼️ Inference

After training, the model generates reports for random samples:

```bash
# Inference preview
python train.py
```

Example output:
```
📸 1225_IM-0150-1001.dcm.png
✅ Actual   : No acute abnormality identified.
🤖 Generated: No acute cardiopulmonary findings.
```

---

## 🔍 Example Results

| Image | Ground Truth | Generated |
|-------|--------------|-----------|
| ![](sample1.png) | No acute abnormality. | No acute disease. |
| ![](sample2.png) | Clear lungs. | Lungs are clear. |

---

## 📂 Files

| File | Description |
|------|-------------|
| `train.py` | Complete training & inference pipeline |
| `model/` | Saved model weights and tokenizer files |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

---

## 🔗 Model on Hugging Face

Model hosted at:  
👉 [RakeshNJ12345/r2Gen-Radiology](https://huggingface.co/RakeshNJ12345/r2Gen-Radiology)

---

## 🧠 Future Improvements

- Try different decoders (T5, BART)
