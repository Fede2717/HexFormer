## Running HexFormer

Follow these steps to set up and run HexFormer:

1. **Create a new environment**  
   The original code was tested with **Python 3.12**.  

2. **Activate the environment and install the requirements**  
    ```bash
    pip install -r requirements.txt
    ```
3. **Choose a configuration file**  
The `config` folder contains configuration files for HexFormer HexFormer-Hybrid and Euclidean-ViT.  
    - Pick one and edit it as needed.  
    - Copy the first commented line in the config file and run it (this is the command to run).  

**Example:**  
To run with `HexFormer.txt` config, use:  
```bash
python classification_vit/train.py -c classification_vit/config/HexFormer.txt
```
