cd ..
call conda activate easygsplat
set KMP_DUPLICATE_LIB_OK=TRUE

@REM python gaussian_viewer.py --path="./data/tandt_db/playroom/"  --gs="./data/pretrained/playroom/final.npy"

python gaussian_viewer.py --path="D:/GS-project/Z-anatomy/gsdata/zanatomy-muscular" --gs="D:/GS-project/MultiLayer-3DGS/Multi-Layer-Anatomy-GS-Training/eval/zanatomy-muscular/point_cloud/iteration_30000/point_cloud.ply"

pause