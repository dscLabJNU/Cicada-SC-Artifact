### Modify TorchVision Models

1. **Clone the TorchVision Repository**  
   Run the following command to clone the official TorchVision repository:  
   ```bash
   git clone https://github.com/pytorch/vision.git
   ```

2. **Checkout a Specific Commit**  
   Navigate into the cloned `vision` directory and checkout the following commit to ensure compatibility:  
   ```bash
   git checkout 06a925c32b49fd0455e265097fa7ca623cec4154
   ```
   > Commit Info:  
   > `fix: fix code block rendering problem in 'Using models from Hub' (#8846)`  
   > Author: GdoongMathew  
   > Date: Mon Jan 13 18:37:55 2025 +0800

3. **Replace Model Files**  
   Copy the following model files from current folder into the corresponding directory inside the cloned TorchVision repo:  
   - `resnet.py`  
   - `vgg.py`  
   - `vision_transformer.py`  

   Replace the originals located in:  
   ```bash
   ./vision/torchvision/models/
   ```

4. **Install TorchVision in Editable Mode**  
   After applying the changes, run the following command to install the modified version of TorchVision:  
   ```bash
   pip install -e .
   ```