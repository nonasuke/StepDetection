import cv2
import torch
import numpy as np
from edge_detection_cnn import EDC
import torch.nn as nn
from natsort import natsorted
import glob


def makeEdgeImg(net, device="cuda:0"):

    num = 3
    filename = str(num) + ".png"
    img = cv2.imread("input/rgb/" + filename)
    d_img = cv2.imread("input/depth/" + filename, cv2.IMREAD_ANYDEPTH)
    h, w, c = img.shape

    mean = [119.0381, 131.4019, 137.0342, 2950.4902]
    std = [52.4009, 49.7220, 50.3291, 975.3069]

    rgbd = np.zeros((h, w, 4))
    rgbd[:, :, :3] = img
    rgbd[:, :, 3] = d_img
    rgbd = (rgbd - mean) / std
    print(rgbd)

    stride = 1
    patch_size = 64
    img_tensor = torch.from_numpy(rgbd).float()

    patches = img_tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    p_h_num, p_w_num = patches.size(0), patches.size(1)
    print(p_h_num, p_w_num)
    data = patches.reshape(patches.size(0)*patches.size(1), 4, patch_size, patch_size)  # flow ->
    l = data.size(0)
    batch_size = 100
    num = int(l / batch_size)

    with torch.no_grad():

        img = torch.from_numpy(img).to(device)
        point = 1

        for i in range(num):
            s_p = i*batch_size
            # sub_data = data[s_p:s_p + batch_size+1]
            sub_data = data[s_p:s_p + batch_size]
            sub_data = sub_data.to(device)

            out = net(sub_data)

            y = nn.functional.softmax(out)
            y_pred_prob = nn.functional.softmax(out)

            y[:, 1] = 0.
            y_pred_prob = torch.where(y_pred_prob > 0.95, y_pred_prob, y)
            # 予測したラベルを予測確率から計算
            y_pred_label = torch.max(y_pred_prob, 1)[1].reshape(sub_data.size(0), 1)

            for label in y_pred_label:

                x_p = point % p_w_num
                y_p = int(point / p_w_num)
                point += 1

                if label[0] == 0:
                    continue
                else:
                    x_s = x_p*stride
                    # x_e = x_s + patch_size
                    y_s = y_p*stride
                    # y_e = y_s + patch_size
                    a = int(patch_size/2)-1
                    b = int(patch_size/2)+1
                    img[y_s+a:y_s+b, x_s+a:x_s+b, :] = torch.tensor([0, 0, 255])

        img = img.cpu().detach().numpy()

        # cv2.imwrite("result/" + filename, img)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def makeEdgeImg2(net, device="cuda:0"):

    rgb_path = natsorted(glob.glob("input_patch_2/64/rgb/*.png"))
    depth_path = natsorted(glob.glob("input_patch_2/64/depth/*.png"))
    patch_size = 64
    rgbd_list = []
    mean = [119.0381,  131.4019,  137.0342, 2950.4902]
    std = [52.4009,  49.7220,  50.3291, 975.3069]

    for i, [rgb, depth] in enumerate(zip(rgb_path, depth_path)):
        if (i >= 279):

            rgb_img = cv2.imread(rgb)
            # cv2.imshow("rgb_patch", rgb_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            d_img = cv2.imread(depth, cv2.IMREAD_ANYDEPTH)
            h, w, c = rgb_img.shape
            rgbd = np.zeros((h, w, 4))
            rgbd[:, :, :3] = rgb_img
            rgbd[:, :, 3] = d_img
            rgbd = (rgbd - mean) / std
            rgbd_list.append(rgbd)

    data = torch.tensor(rgbd_list).float()

    data = data.reshape(-1, 4, patch_size, patch_size)  # flow ->

    with torch.no_grad():

        data = data.to(device)

        out = net(data)

        y_pred_prob = nn.functional.softmax(out)

        # y[:, 1] = 0.
        # print(y_pred_prob)
        # y_pred_prob = torch.where(y_pred_prob > 0.8, y_pred_prob, y)
        # print(y_pred_prob)
        # 予測したラベルを予測確率から計算
        y_pred_label = torch.max(y_pred_prob, 1)[1].reshape(data.size(0), 1)

        print(y_pred_label)


if __name__ == "__main__":

    model = EDC(patch_size=64)
    model_path = 'good_model_gpu.pth'
    model.load_state_dict(torch.load(model_path))

    makeEdgeImg(model.to('cuda:0'))
