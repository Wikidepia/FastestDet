import argparse
import os
import time

import cv2
import numpy as np
import onnx
import torch
from onnxsim import simplify

from module.detector import Detector
from utils.tool import *


# https://stackoverflow.com/a/44659589
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def pad_to_square(img, size, pad_value=0):
    img = image_resize(img, width=size)
    # pad to sizexsize
    h, w, _ = img.shape
    h_diff = size - h
    w_diff = size - w
    # keep image in top left
    bottom = h_diff
    right = w_diff
    img = cv2.copyMakeBorder(
        img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=pad_value
    )
    return img


def restore_coords(boxes, orig_shape, pad_shape):
    orig_h, orig_w = orig_shape
    pad_h, pad_w = pad_shape

    # Rescale coordinates to original dimensions
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * pad_w).clip(0, orig_w)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * pad_h).clip(0, orig_h)

    return boxes


if __name__ == "__main__":
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="", help=".yaml config")
    parser.add_argument("--weight", type=str, default=None, help=".weight config")
    parser.add_argument("--img", type=str, default="", help="The path of test image")
    parser.add_argument(
        "--thresh", type=float, default=0.65, help="The path of test image"
    )
    parser.add_argument(
        "--onnx", action="store_true", default=False, help="Export onnx file"
    )
    parser.add_argument(
        "--torchscript",
        action="store_true",
        default=False,
        help="Export torchscript file",
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Run on cpu")

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 选择推理后端
    if opt.cpu:
        print("run on cpu...")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("run on gpu...")
            device = torch.device("cuda")
        else:
            print("run on cpu...")
            device = torch.device("cpu")

    # 解析yaml配置文件
    cfg = LoadYaml(opt.yaml)
    print(cfg)

    # 模型加载
    print("load weight from:%s" % opt.weight)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    # sets the module in eval node
    model.eval()

    # 数据预处理
    ori_img = cv2.imread(opt.img)
    res_img = pad_to_square(ori_img, cfg.input_height)
    ret_img = res_img.copy()
    img = res_img.reshape(cfg.input_height, cfg.input_width, 3)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    img = img.unsqueeze(0)
    img = img.to(device).float() / 255.0

    # 导出onnx模型
    if opt.onnx:
        torch.onnx.export(
            model,  # model being run
            img,  # model input (or a tuple for multiple inputs)
            "./FastestDet.onnx",  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,
        )  # whether to execute constant folding for optimization
        # onnx-sim
        onnx_model = onnx.load("./FastestDet.onnx")  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        print("onnx sim sucess...")
        onnx.save(model_simp, "./FastestDet.onnx")

    # 导出torchscript模型
    if opt.torchscript:
        import copy

        model_cpu = copy.deepcopy(model).cpu()
        x = torch.rand(1, 3, cfg.input_height, cfg.input_width)
        mod = torch.jit.trace(model_cpu, x)
        mod.save("./FastestDet.pt")
        print(
            "to convert torchscript to pnnx/ncnn: ./pnnx FastestDet.pt inputshape=[1,3,%d,%d]"
            % (cfg.input_height, cfg.input_height)
        )

    # 模型推理
    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    time = (end - start) * 1000.0
    print("forward time:%fms" % time)

    # 特征图后处理
    output = handle_preds(preds, device, opt.thresh, nms_thresh=0.1)

    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, "r") as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    H, W, _ = cfg.input_height, cfg.input_width, 3
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

    # 绘制预测框
    for box in output[0]:
        print(box)
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ret_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ret_img, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ret_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.png", ret_img)
