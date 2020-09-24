import torch.optim as optim
from graphics import *
from net import *
from utils import *
from dataset import *

sselector = TSampleSelector('../Furniture1')
training_df = sselector.sampling()
img_folder = '../Furniture1'
tdataset = TrainingDataset(training_df, img_folder)
train_loader = DataLoader(tdataset, batch_size=1, shuffle=False)
model = Net().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# for simplicity, defaultly the batch size should be set to 1

for epoch in range(100):
    for data in train_loader:

        #clear the grad:
        optimizer.zero_grad()
        # read input shape:
        # data[0]: data input after plane sweep, which directly input to the network
        # data[1]: target x
        # data[0] input shape: N(1), c, h, w where N represents the number of records(One source, one target), default=1
        net_input = data[0]
        source_x = int(data[1])
        target_x = int(data[2])

        # read the target image: 
        target_img = read_raw_tensorimg(img_folder, target_x)
        target_img = np.transpose(target_img, (2, 0, 1))
        target_img = torch.from_numpy(target_img).cuda().float()
        
        # target_img shape: (c, h, w)
        net_input = net_input.float().cuda()
        net_output = model(net_input) # output shape: 1, c, h, w

        # get mpi representation:
        color_imgs, alpha_imgs, reference_x = get_mpi_singleinput(net_output, source_x)
        novel_view = translation_view_render(color_imgs, alpha_imgs, reference_x, target_x)

        # calculate the mse loss to the result and update the parameters:
        loss = ((novel_view - target_img) ** 2).flatten().sum()
        loss.backward()
        optimizer.step()
        
        # post processing: save model and sample fig:
        print('loss is: ', loss)
        #torch.save(model.state_dict(), 'model.pth')
        #cv2.imwrite("sample.png", np.transpose((novel_view * 255).clone().cpu().detach().numpy(), (1, 2, 0)))