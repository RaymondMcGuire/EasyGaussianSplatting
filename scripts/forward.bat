cd ..
call conda activate easygsplat
set KMP_DUPLICATE_LIB_OK=TRUE

python forward_gpu.py --gs="./data/final.npy"

pause