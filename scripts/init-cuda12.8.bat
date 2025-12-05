call conda env create -f environment.yml

call conda activate easygsplat

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install ./gsplatcu --no-build-isolation
pause