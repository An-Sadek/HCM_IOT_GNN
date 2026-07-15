torchrun --standalone --nproc_per_node=gpu train/run_train.py
conda create --name dgl python=3.12
conda activate dgl
conda install -c dglteam/label/th24_cu124 dgl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 pyyaml scikit-learn transformers
