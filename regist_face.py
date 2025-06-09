import os, yaml, numpy as np, pickle as pkl, torch
from PIL import Image

from models import initialize_mtcnn, initialize_arcface, BACKBONE
from tools import  normalize_point, align_face, img2tensor, crop_face, REFERENCE_POINT

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

facebank_dir = {
    "processed": os.path.join(os.getcwd(), CONFIG["facebank_dir"], "processed"),
    "unprocessed": os.path.join(os.getcwd(), CONFIG["facebank_dir"], "unprocessed")
}


mtcnn = initialize_mtcnn(device=device, selection_method="largest")
arcface = initialize_arcface(device, bbone=BACKBONE.IRESNET100)

for i in os.listdir(facebank_dir["unprocessed"]):
    i_path = os.path.join(facebank_dir["unprocessed"], i)
    i_proc_path = os.path.join(facebank_dir["processed"], i)

    os.makedirs(i_proc_path, exist_ok=True)

    for j in os.listdir(i_path):
        if j.endswith((".png", ".jpg")):
            img = Image.open(os.path.join(i_path, j)).convert("RGB")
            img = np.array(img)
            detect = mtcnn.detect(img,landmarks=True)

            if detect[0] is None:
                print(f"{j} NOT FOUND")
                continue

            if detect[1][0] < 0.97:
                print(f"{j} NOT FOUND")
                continue

            cropped_face = crop_face(img, detect[0][0],convert_rgbobgr=False)
            normalized_point = normalize_point(detect[2][0], detect[0][0])
            # rs_img, rs_point = resize_scale(cropped_face, normalized_point)
            aligned_face = align_face(cropped_face, normalized_point,REFERENCE_POINT)
            
            final_img = Image.fromarray(aligned_face)
            final_img.save(os.path.join(i_proc_path, j))


input("Tekan Enter untuk mulai memasukkan data")

results = {
    "features": [],
    "classes": []
}

for i in os.listdir(facebank_dir["processed"]):
    i_path = os.path.join(facebank_dir["processed"], i)
    success_cnt = 0
    for j in os.listdir(i_path):
        if j.endswith((".png", ".jpg")):
            img = Image.open(os.path.join(i_path, j)).convert("RGB")
            img = np.array(img)
            
            tensor_face = img2tensor(img, device=device)
            with torch.no_grad():
                features = arcface(tensor_face)[0].detach().cpu()
                results["features"].append(features)
                results["classes"].append(i)
                success_cnt+=1

    print(f"{i} got {success_cnt} image extracted")

assert len(results["features"]) > 0, "No image found!"
results["features"] = torch.stack(results["features"])
print(results["features"].shape)

with open(CONFIG["facebank_output"], "wb") as output:
    pkl.dump(results, output)
    print("Success")

