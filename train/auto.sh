#torchrun --standalone --nproc_per_node=gpu train/run_train.py --epochs 100 --batch-size 32 --hidden-dim 16 --lr 0.01
cd ~
git clone --recurse-submodules https://github.com/dmlc/dgl.git
sudo apt-get update
sudo apt-get install -y build-essential python3-dev make cmake
cd ~/dgl
bash script/create_dev_conda_env.sh -g 12.1

conda activate dgl-dev-gpu-121
CONDA_NO_PLUGINS=true conda install -y -c dglteam/label/th21_cu121 dgl
pip install "transformers==4.45.2" "tokenizers<0.21" accelerate safetensors sentencepiece protobuf 
pip install pyyaml pandas scikit-learn "numpy<2"

cd ~
git clone https://github.com/An-Sadek/HCM_IOT_GNN.git
cd HCM_IOT_GNN
python preprocess/pipelines/run_pipelines.py
python train/run_train.py

