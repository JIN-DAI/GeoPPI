
# re-download a large trained GBT file
if [ ! -f "trainedmodels/gbt-s4169.pkl" ];
then
	curl -OL https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl
	mv gbt-s4169.pkl trainedmodels/gbt-s4169.pkl
fi

# build environment
source activate
conda create -n ppi python==3.8.5 -y
conda activate ppi

# dependencies
# torch==1.7.0+cpu -f  https://download.pytorch.org/whl/torch_stable.html
conda install -c pytorch pytorch==1.7.0 

# torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html  
curl -OL https://data.pyg.org/whl/torch-1.7.0%2Bcpu/torch_cluster-1.5.8-cp38-cp38-macosx_10_9_x86_64.whl
pip install "torch_cluster-1.5.8-cp38-cp38-macosx_10_9_x86_64.whl"

# torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
curl -OL https://data.pyg.org/whl/torch-1.7.0%2Bcpu/torch_scatter-2.0.5-cp38-cp38-macosx_10_9_x86_64.whl
pip install "torch_scatter-2.0.5-cp38-cp38-macosx_10_9_x86_64.whl"

# torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
curl -OL https://data.pyg.org/whl/torch-1.7.0%2Bcpu/torch_sparse-0.6.8-cp38-cp38-macosx_10_9_x86_64.whl
pip install "torch_sparse-0.6.8-cp38-cp38-macosx_10_9_x86_64.whl"

# torch-geometric==1.4.1
pip install torch-geometric==1.4.1

# scikit-learn==0.24.1
pip install scikit-learn==0.24.1

# pymol
conda install -c schrodinger pymol -y  # need license

# clean
rm ./*.whl
