# 🎭 FESTA Framework Demo

This directory contains the Google Colab demo for the **FESTA** approach for measuring uncertainty in Multimodal Large Language Models (MLLMs).

- `FESTA_Unsloth_Demo.ipynb` - Main Colab notebook with dataset examples
- `examples_config.json` - Configuration file with example specifications  
- `examples/` - Directory containing selected BLINK dataset images

## 🖼️ Included Dataset Examples

### From BLINK Dataset:
1. **val_Spatial_Relation_1.jpg** + perturbations - Cat and car spatial relationships
2. **val_Spatial_Relation_10.jpg** + perturbations - Multi-object scenes  
3. **val_Spatial_Relation_25.jpg** + perturbations - Animal detection scenarios
4. **val_Spatial_Relation_50.jpg** + perturbations - Indoor/outdoor classification
5. **val_Spatial_Relation_75.jpg** + perturbations - Subject identification with masking

### Perturbation Types:
- **blur1** - Gaussian blur for visual degradation
- **contrast1** - Reduced contrast for visibility testing  
- **noise1** - Added noise for robustness evaluation
- **bw1** - Black-white conversion for color dependency
- **masking1** - Partial occlusion for attention testing

## 🧪 Demo Structure

### Equivalent Examples (5 cases):
- Test consistency when same semantic meaning, different phrasing/perturbation
- Includes 1 expected failure case (masking perturbation)
- Covers: spatial relationships, object detection, animal identification, subject recognition

### Complementary Examples (5 cases):  
- Test opposite answers for semantically opposite questions
- Includes 1 expected failure case (color properties)
- Covers: spatial opposites, presence/absence, indoor/outdoor, distance relationships

## 🚀 Usage Instructions

### For Google Colab:
1. Upload `FESTA_Demo_Dataset.ipynb` to Google Colab
2. Make sure to upload the `examples/` folder with all images to your Colab environment  
3. Run all cells sequentially
4. The notebook will automatically:
   - Install required packages
   - Load the LLaVA model
   - Run all demo examples
   - Generate summary visualizations

### For GitHub Repository:
1. Copy all files to the `examples/` directory in your repository:
   ```
   examples/
   ├── FESTA_Demo_Dataset.ipynb
   ├── demo_examples.json  
   ├── README.md
   └── images/
       ├── val_Spatial_Relation_1.jpg
       ├── val_Spatial_Relation_1_blur1.jpg
       ├── val_Spatial_Relation_10.jpg
       ├── val_Spatial_Relation_10_contrast1.jpg
       ├── ... (all other example images)
   ```

2. Update the notebook's download URLs to point to your repository:
   ```python
   base_url = "https://raw.githubusercontent.com/iiscleap/mllm-uncertainty-estimation/main/examples/images/"
   ```

## 📊 Expected Results

The demo is designed to showcase:
- **Success cases**: Where FESTA correctly identifies model consistency
- **Failure cases**: Where FESTA reveals model inconsistencies  
- **Real-world applicability**: Using actual dataset examples vs synthetic ones
- **Systematic evaluation**: Structured approach to uncertainty measurement

## 🎯 Key Features

- ✅ **Real dataset examples** from BLINK and VSR
- ✅ **Interactive visualizations** with success/failure indicators
- ✅ **Comprehensive analysis** with summary statistics
- ✅ **Educational content** explaining FESTA methodology
- ✅ **Reproducible results** with fixed model configurations

## 🤝 Contributing

To improve this demo:
1. Add more diverse dataset examples
2. Include additional perturbation types
3. Implement more sophisticated consistency metrics
4. Add support for other LVLM models
5. Create failure case analysis tools

<!-- ## 📚 Citation

If you use this demo in your research, please cite:

```bibtex
@article{festa2024,
  title={FESTA: Framework for Evaluating Semantic and Temporal Assumptions in Multimodal LLMs},
  author={[Authors]},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2024}
} -->
```

