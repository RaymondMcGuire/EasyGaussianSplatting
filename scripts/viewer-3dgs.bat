cd ..
call conda activate easygsplat
set KMP_DUPLICATE_LIB_OK=TRUE

python gaussian_viewer.py --gs="D:/unity-project/library/VR-GS/VR-GS/Data/dance_siyu/_0_point_cloud.ply"

REM python gaussian_viewer.py --gs="D:/GS-project/MultiLayer-3DGS/Multi-Layer-Anatomy-GS-Training/eval/zanatomy-muscular/point_cloud/iteration_30000/point_cloud.ply"

pause