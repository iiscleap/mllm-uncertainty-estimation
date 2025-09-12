# FESTA Demo - Google Colab Integration

## 🚀 Quick Start

Click this button to run the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iiscleap/mllm-uncertainty-estimation/blob/main/festa_demo/FESTA_Simple_Demo.ipynb)

## 📋 How it Works

1. **Click the "Open in Colab" button** above or in the notebook
2. **Google Colab will open** the notebook directly from your GitHub repository
3. **Run all cells** to load LLaVA 1.6 7B model and test with FESTA examples
4. **Images are automatically downloaded** from your GitHub repo

## 🔧 Setting Up the Colab Button

The Colab button uses this URL format:
```
https://colab.research.google.com/github/{username}/{repo}/blob/{branch}/{path_to_notebook}
```

For your repository:
```
https://colab.research.google.com/github/iiscleap/mllm-uncertainty-estimation/blob/main/festa_demo/FESTA_Simple_Demo.ipynb
```

## 📁 Required Files on GitHub

Make sure these files are in your repository:

```
festa_demo/
├── FESTA_Simple_Demo.ipynb  (with Colab badge)
└── examples/
    ├── val_Spatial_Relation_1.jpg
    ├── val_Spatial_Relation_1_contrast1.jpg  
    ├── val_Spatial_Relation_1_masking1.jpg
    ├── val_Spatial_Relation_1_negated_contrast1.jpg
    ├── val_Spatial_Relation_5.jpg
    └── val_Spatial_Relation_5_blur1.jpg
```

## 🎯 User Experience

Users can:
1. Click the badge on GitHub
2. Notebook opens in Colab automatically  
3. Run cells to load model and test images
4. Modify questions or add custom tests
5. See visual results with image + LLM response

No manual file uploads needed - everything loads from GitHub!
