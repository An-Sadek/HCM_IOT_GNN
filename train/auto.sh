#torchrun --standalone --nproc_per_node=gpu train/run_train.py
conda create -y --name dgl python=3.10

conda activate dgl

conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

conda install -y -c dglteam/label/th24_cu124 dgl

conda install -y pandas pyyaml scikit-learn pydantic

python -m pip install "transformers==4.45.2" "tokenizers<0.21" accelerate safetensors sentencepiece protobuf

cd ~

git clone https://github.com/An-Sadek/HCM_IOT_GNN.git
cd HCM_IOT_GNN
python preprocess/pipelines/run_pipelines.
python train/run_train.py

