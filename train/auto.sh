#torchrun --standalone --nproc_per_node=gpu train/run_train.py
conda create -y --name dgl python=3.12
conda activate dgl
conda install -y -c dglteam/label/th24_cu124 dgl
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c dglteam/label/th24_cu124 dgl
