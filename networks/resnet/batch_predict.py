import os
import json

import torch
from torch import nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "./honor/val/black"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    print(len(img_path_list))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)


    # create model
    model = resnet34(num_classes=3).to(device)

    # load model weights
    weights_path = './models/12-3-21-08/resNet34.pth'
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    # with torch.no_grad():
    #     for ids in range(0, len(img_path_list) // batch_size):
    #         img_list = []
    #         for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
    #             assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
    #             img = Image.open(img_path)
    #             img = data_transform(img)
    #             img_list.append(img)
    #
    #         # batch img
    #         # 将img_list列表中的所有图像打包成一个batch
    #         batch_img = torch.stack(img_list, dim=0)
    #         # predict class
    #         output = model(batch_img.to(device)).cpu()
    #         predict = torch.softmax(output, dim=1)
    #         probs, classes = torch.max(predict, dim=1)
    #
    #         for idx, (pro, cla) in enumerate(zip(probs, classes)):
    #             print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
    #                                                              class_indict[str(cla.numpy())],
    #                                                              pro.numpy()))

    # 验证
    path = './'
    valdir = os.path.join(path, 'honor/val')
    valid_set = datasets.ImageFolder(valdir, transform=data_transform)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    dataset_size = len(valid_set)

    running_loss = 0.0
    running_corrects = 0
    for (inputs, labels) in valid_loader:
        # 输入的属性
        inputs = Variable(inputs.to(device))
        # 标签
        labels = Variable(labels.to(device))
        # 预测
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # 计算损失
        loss = criterion(outputs, labels)

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    print("epoch_loss:" + str(epoch_loss))
    print("epoch_acc:" + str(epoch_acc))

if __name__ == '__main__':
    main()
