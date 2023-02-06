import torch



print("Torch cuda version",torch.version.cuda)
print(f"Torch version: {torch.__version__}") 
print(f"Cuda available: {torch.cuda.is_available()}") 

import torch_geometric
print(f"Torch geometric version: {torch_geometric.__version__}") 