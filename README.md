# CIS 583 Project

## Requirements
1. Windows 10/11.
2. NVIDIA GPU (Tested on NVIDIA GeForce RTX 3080 Laptop GPU and NVIDIA GeForce RTX 4090 GPU)
3. WSL 2.
4. Installed nvidia-toolkit-container and Docker on WSL 2 using 
https://docs.nvidia.com/cuda/wsl-user-guide/index.html and
https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl.
5. Python, matplotlib, pandas, and numpy installed on Windows for data preprocessing and visualization. 

## Running the Project
Once the project requirements have been satisfied, enter the root project directory.     
Run ```python .\bin\clean_data.py``` to prepare the dataset.        
Enter WSL 2 and run ```make``` to automatically build and run the image.        
To visualize metrics from a run, run ```python .\bin\visualize_data.py .\raw_results\raw_results_file.csv```.

## Project Capabilities
Once the program is running and a prompt is given, type ```t``` to train and save a model.    
If you've already trained a model and would like to evaluate it with evaluation data, type ```e```.    
If you've already trained a model and would like to classify your own data with it,
make sure you have a CSV in the same format as test/data/hulk2.csv or test/data/slowloris3.csv and type ```c```.

## Programming References
Some references were used to aid in setting up the environment for our project and as aids in our programming:
1. https://docs.nvidia.com/cuda/wsl-user-guide/index.html
2. https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/
3. https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl
4. https://geeksforgeeks.org/data-normalization-with-pandas/
5. https://pytorch.org/docs/stable/