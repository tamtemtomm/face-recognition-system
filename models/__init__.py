from .arcface import *
from .mtcnn import *

import gdown

import yaml
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

from types import SimpleNamespace
BACKBONE = SimpleNamespace(
    IRESNET100 = 1,
    RESNET50 = 2
)

def initialize_arcface(device="cpu", bbone:SimpleNamespace=BACKBONE.RESNET50) -> Backbone | IResNet100:
    assert bbone in (1, 2)

    if bbone == BACKBONE.RESNET50:
        mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/arcface/ir50.pth")
        if not os.path.exists(mpath):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/arcface"), exist_ok=True)
            gdown.download(id=CONFIG['resnet_50_gdownload_id'], output=mpath)

        print("Using Resnet50")
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        model.load_state_dict(torch.load(mpath, weights_only=True, map_location=torch.device(device)))
    elif bbone == BACKBONE.IRESNET100:
        mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/arcface/ir100.pt")
        print("Using IResNet100")
        if not os.path.exists(mpath):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/arcface"), exist_ok=True)
            gdown.download(id=CONFIG['resnet_100_gdownload_id'], output=mpath)

        model = IResNet100(True, device=device, pretrain_path=mpath)

    model.eval()
    
    return model

def initialize_mtcnn(device="cpu", **kwargs) -> MTCNN:
    mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/mtcnn")
    print(all([x in os.listdir(mpath) for x in ["onet.pt", "pnet.pt", "rnet.pt"]]))
    if not all([x in os.listdir(mpath) for x in ["onet.pt", "pnet.pt", "rnet.pt"]]):
        os.makedirs(mpath, exist_ok=True)
        gdown.download_folder(id=CONFIG['mtcnn_gdownload_id'], output=mpath)
    model = MTCNN(device=device, **kwargs)
    return model