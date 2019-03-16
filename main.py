import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
#from torchsummary import summary
import cv2
import scipy.misc

# 是否使用cuda
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" )

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
num_epochs=10

l = []
def train_model(model, criterion, optimizer, dataload):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            print (outputs.shape)
            print (labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        l.append(epoch_loss)
        np.save('loss',np.array(l))
    torch.save(model.state_dict(), 'weight/weights_%d.pth' % epoch)
    return model

#训练模型
def train():
    model = Unet(3, 1).to(device)
    #summary(model,(3,512,512))
    batch_size = 1
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/image","data/mask",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test():
    model = Unet(3, 1)
    model.load_state_dict(torch.load('weight/weights_{}.pth'.format(str(num_epochs-1)),map_location='cpu'))
    liver_dataset = LiverDataset("data/img","data/label",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    i = 0
    with torch.no_grad():
        for x, _ in dataloaders:
            #print (x.shape)
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            print (img_y)
            img = cv2.normalize(img_y,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

            cv2.imwrite('data/pred/{}.png'.format(str(i)),img)

            i = i+1
            print (i)



if __name__ == '__main__':

    train()
    test()