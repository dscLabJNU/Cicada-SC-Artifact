### Instructions for Running Experiments

1. **Navigate to the Project Directory**  
   Go to the `pytorch_files` folder and apply the necessary changes to the **PyTorch/Vision** submodules as required.

2. **Install Dependencies**  
   Run the following command to install all required Python packages:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Azure Dataset**  
   - Download the dataset from the following link:  
     [Azure Functions Invocation Trace 2021](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md)
   - After downloading, update the dataset path in the following scripts:  
     - `model_load_tetris.py`: update the `--azure-trace` argument in the `parse_arguments()` function  
     - `pipeline_model.py`: update the `azure_trace_path` in the `main()` function

4. **Run Initial Experiments**  
   Execute the following script to run the primary experiments and generate initial results:  
   ```bash
   bash run_experiments.sh
   ```

5. **Run Memory Comparison Experiments**  
   To evaluate memory usage, run the following script:  
   ```bash
   bash run_memory_comparison.sh
   ```

6. **Access and Analyze Results**  
   - All experiment results will be saved to the `./logs` directory.  
   - To change the analysis target, edit the `main()` function in `analyze_separate_structure.py`.

7. **Explore Motivating Experiments**  
   If you are interested in our motivating experiments, navigate to the `./analyze` folder and refer to the included `README.md`.