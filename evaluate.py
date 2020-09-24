from utils import *
from dataset import *
from net import *

torch.cuda.empty_cache()
torch.cuda.set_device(1)

sselector = TSampleSelector('../Furniture3')
training_df = sselector.sampling()
img_folder = '../Furniture3'
tdataset = TrainingDataset(training_df, img_folder)
train_loader = DataLoader(tdataset, batch_size=1, shuffle=False)
model = Net().cuda()
model.eval()
torch.no_grad()
model.load_state_dict(torch.load('model.pth'))
model = model.cpu()

losses = []

for i, data in enumerate(train_loader):

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
    target_img = torch.from_numpy(target_img).float()#.cuda().float()

    # target_img shape: (c, h, w)
    net_input = net_input.float()#.cuda()
    net_output = model(net_input) # output shape: 1, c, h, w

    # get mpi representation:
    color_imgs, alpha_imgs, reference_x = get_mpi_singleinput(net_output, source_x)
    novel_view = translation_view_render(color_imgs, alpha_imgs, reference_x, target_x)

    # calculate the mse loss to the result and update the parameters:
    loss = ((novel_view - target_img) ** 2).flatten().sum()

    #print('loss is: ', loss)
    losses.append(int(loss.detach().numpy()))
    cv2.imwrite("evaluate/evaluate_sample"+ str(i) + ".png", np.transpose((novel_view * 255).clone().detach().numpy(), (1, 2, 0)))
    print(loss)

print('final average loss:', sum(losses) / len(losses))