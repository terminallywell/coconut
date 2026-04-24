# verify GPU
nvidia-smi



# clone & checkout branch
git clone https://github.com/terminallywell/coconut.git
cd coconut
git checkout dev

# create & setup conda environment
conda create --name coconut python=3.12 -y
conda activate coconut
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# log in to wandb
wandb login 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


