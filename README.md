# SeemoRe ONNX Model

This repository contains the ONNX conversion of the SeemoRe model for efficient image super-resolution.

## Original Model Information

**Paper**: See More Details: Efficient Image Super-Resolution by Experts Mining  
**Conference**: ICML 2024  
#### [Eduard Zamfir<sup>1</sup>](https://eduardzamfir.github.io), [Zongwei Wu<sup>1*</sup>](https://sites.google.com/view/zwwu/accueil), [Nancy Mehta<sup>1</sup>](https://scholar.google.com/citations?user=WwdYdlUAAAAJ&hl=en&oi=ao),  [Yulun Zhang<sup>2,3*</sup>](http://yulunzhang.com/) and [Radu Timofte<sup>1</sup>](https://www.informatik.uni-wuerzburg.de/computervision/)

#### **<sup>1</sup> University of Würzburg, Germany - <sup>2</sup> Shanghai Jiao Tong University, China - <sup>3</sup> ETH Zürich, Switzerland**
#### **<sup>*</sup> Corresponding authors**

### Links
- **Original Repository**: https://github.com/eduardzamfir/seemoredetails
- **Paper**: [ArXiv](https://arxiv.org/abs/2402.03412) 
- **Project Page**: Available in original repository
- **Pre-trained Model**: [Google Drive](https://drive.google.com/drive/folders/1qaGi2Oi1GsgFb2T5BoBU5GCWsHicrQLK)


## Prerequisites

To export the model to ONNX format, you need:

```bash
pip install torch torchvision onnx
```

### Required Dependencies
- Python 3.7+
- PyTorch
- ONNX
- The original SeemoRe model implementation

## ONNX Export Process

### Step 1: Download the Pre-trained Model
Download `net_g_latest.pth` from the [Google Drive link](https://drive.google.com/drive/folders/1qaGi2Oi1GsgFb2T5BoBU5GCWsHicrQLK) provided by the original authors.

### Step 2: Conversion Script

```python
import torch
from basicsr.archs.seemore_arch import SeemoRe  # Import from original repo

# === Step 1: Reconstruct the Model ===
model = SeemoRe(
    scale=4,
    in_chans=3,
    num_experts=3,
    num_layers=6,
    embedding_dim=48,
    img_range=1.0,
    use_shuffle=True,
    lr_space='exp',
    topk=1,
    recursive=2,
    global_kernel_size=11
)

# === Step 2: Load Checkpoint ===
ckpt = torch.load('net_g_latest.pth', map_location='cpu')
model.load_state_dict(ckpt['params'], strict=False)
model.eval()

# === Step 3: Export to ONNX ===
dummy_input = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model,
    dummy_input,
    'seemore_x4v2_static512.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None
)

```

### Step 3: Run the Conversion

```bash
python convert_to_onnx.py
```

This will generate `seemore_x4v2_static512.onnx` file ready for use with OpenCV DNN or other ONNX-compatible frameworks.

## License

This model conversion follows the original repository's licensing:

### Apache License 2.0
The original SeemoRe model is licensed under the Apache License 2.0. This is a permissive license whose main conditions require preservation of copyright and license notices.

**Permissions:**
- Commercial use
- Modification  
- Distribution
- Patent use
- Private use

**Conditions:**
- License and copyright notice
- State changes

**Limitations:**
- Trademark use
- Liability
- Warranty

For full license details, see the [LICENSE file](https://github.com/eduardzamfir/seemoredetails/blob/main/LICENSE) in the original repository.

## Credits

This ONNX conversion is based on the original work by:
- Eduard Zamfir (University of Würzburg, Germany)
- Zongwei Wu* (University of Würzburg, Germany)
- Nancy Mehta (University of Würzburg, Germany)  
- Yulun Zhang* (Shanghai Jiao Tong University, China & ETH Zürich, Switzerland)
- Radu Timofte (University of Würzburg, Germany)

Please cite the original paper if you use this model:

```bibtex
@inproceedings{zamfir2024seemore,
  title={See More Details: Efficient Image Super-Resolution by Experts Mining},
  author={Zamfir, Eduard and Wu, Zongwei and Mehta, Nancy and Zhang, Yulun and Timofte, Radu},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Acknowledgments

Special thanks to the original authors for providing the pre-trained models and making their work publicly available under an open-source license.
