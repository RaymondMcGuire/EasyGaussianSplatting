from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import re
from gsplat.read_write_model import *
from gsplat.gau_io import load_gs
# from read_write_model import *
from PIL import Image
import torch
import torchvision
from plyfile import PlyData
import torchvision.transforms as transforms
from PIL import Image


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw, path):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw
        self.path = path



class GSplatDataset(Dataset):
    def __init__(self, path, resize_rate=1, device='cuda', gs_path=None) -> None:
        super().__init__()
        self.device = device
        self.resize_rate = resize_rate

        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')
        self.cameras = []
        self.images = []
        for image_param in image_params.values():
            i = image_param.camera_id
            camera_param = camera_params[i]
            im_path = str(Path(path, "images", image_param.name))
            image = Image.open(im_path)
            if (resize_rate != 1):
                image = image.resize((image.width * self.resize_rate, image.height * self.resize_rate))

            w_scale = image.width/camera_param.width
            h_scale = image.height/camera_param.height
            if len(camera_param.params) == 3:
                f = camera_param.params[0]
                fx = f * w_scale
                fy = f * h_scale
                cx = camera_param.params[1] * w_scale
                cy = camera_param.params[2] * h_scale
            else:
                fx = camera_param.params[0] * w_scale
                fy = camera_param.params[1] * h_scale
                cx = camera_param.params[2] * w_scale
                cy = camera_param.params[3] * h_scale
            Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(image_param.tvec).to(self.device).to(torch.float32)
            camera = Camera(image_param.id, image.width, image.height, fx, fy, cx, cy, Rcw, tcw, im_path)
            image = torchvision.transforms.functional.to_tensor(image).to(self.device).to(torch.float32)

            self.cameras.append(camera)
            self.images.append(image)
        self.gs = self._load_gs(Path(path), gs_path)

        twcs = torch.stack([x.twc for x in self.cameras])
        cam_dist = torch.linalg.norm(twcs - torch.mean(twcs, axis=0), axis=1)
        self.sence_size = float(torch.max(cam_dist)) * 1.1

    def __getitem__(self, index: int):
        return self.cameras[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)

    def _load_gs(self, root: Path, gs_path):
        if gs_path:
            return load_gs(gs_path)
        npy_path = Path(root, "sparse/0/points3D.npy")
        if npy_path.exists():
            return np.load(npy_path)
        bin_path = Path(root, "sparse/0/points3D.bin")
        if bin_path.exists():
            gs = read_points_bin_as_gau(bin_path)
            np.save(npy_path, gs)
            return gs
        ply_path = self._find_latest_ply(root)
        if ply_path is not None:
            return load_gs(str(ply_path))
        raise FileNotFoundError("No gs data found under %s" % root)

    def _find_latest_ply(self, root: Path):
        candidates = list(Path(root, "point_cloud").glob("iteration_*/point_cloud.ply"))
        if not candidates:
            return None
        def iter_num(path: Path):
            match = re.search(r"iteration_(\d+)", path.parent.name)
            return int(match.group(1)) if match else -1
        return max(candidates, key=iter_num)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path)
    gs_dataset[0]
