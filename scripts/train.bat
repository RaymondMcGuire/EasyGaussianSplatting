cd ..
call conda activate easygsplat
set KMP_DUPLICATE_LIB_OK=TRUE
python train.py --path="./data/tandt_db/tandt/train"

pause