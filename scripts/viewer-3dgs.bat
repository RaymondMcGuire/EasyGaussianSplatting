cd ..
call conda activate easygsplat
set KMP_DUPLICATE_LIB_OK=TRUE

@REM python gaussian_viewer.py --gs="D:/GS-project/GSModels/bicycle/point_cloud/iteration_30000/point_cloud.ply"

python gaussian_viewer.py --gs="D:/GS-project/MultiLayer-3DGS/Multi-Layer-Anatomy-GS-Training/eval/zanatomy-muscular/point_cloud/iteration_30000/point_cloud.ply"

pause