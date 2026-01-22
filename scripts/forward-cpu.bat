cd ..
call conda activate easygsplat
set KMP_DUPLICATE_LIB_OK=TRUE

python forward_cpu.py --gs="./data/pretrained/playroom/final.npy"

pause