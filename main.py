import torch.optim as optim
from net import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import time
import argparse
import torch.nn as nn
from psnr import PSNR
import torch
from datetime import datetime

parser = argparse.ArgumentParser(description='Some essential arguments')
parser.add_argument('-f', type=float, default=5)
parser.add_argument('--sx', type=float, default=0.01)
parser.add_argument('--sy', type=float, default=0.01)
parser.add_argument('--crop_size', type=int, default=360)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--total_epochs', type=int, default=10)
parser.add_argument('--depth_levels', type=str, default='list(range(20, 100, 5))')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--input_chns', type=int, default=195)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--evaluate', action='store_true', default=None)
parser.add_argument('--shifting_offset', type=float, default=0.021)
parser.add_argument('--fvistrain', action='store_true')
parser.add_argument('--getmpi', action='store_true', help='get the mpi RGBA planes')
parser.add_argument('--project_name', type=str, default=None)
parser.add_argument('--dnet', action="store_true", default=None)

args = parser.parse_args()
check_point = None
#project_name = args.project_name
# some arguments unpacking:
if args.checkpoint_path is not None:
    print('load the checkpoint')
    check_point = torch.load(args.checkpoint_path)
    print('checkpoint parameters:')
    print(check_point['args'])
    print('checkpoint epoch number: ', check_point['epoch'])
    check_point['args'].evaluate = args.evaluate
    check_point['args'].resume = args.resume
    check_point['args'].total_epochs = args.total_epochs
    check_point['args'].batch_size = args.batch_size
    check_point['args'].checkpoint_path = args.checkpoint_path
    check_point['args'].fvistrain = args.fvistrain
    check_point['args'].getmpi = args.getmpi
    check_point['args'].dnet = args.dnet

    if not args.resume:
        project_name = args.project_name
    else:
        project_name = check_point['args'].project_name

    args = check_point['args']
    args.project_name = project_name
    psnr_max = 0#check_point['psnr_max']

if check_point is None:
    psnr_max = 0
depth_levels = eval(args.depth_levels)
graphics = Graphics(f=args.f, sx=args.sx, sy=args.sy, depth_levels=depth_levels, shifting_offset=args.shifting_offset)
# prepare the dataset and other information:
print('prepare the training dataset')
train_dataset = prepareDataset(graphics, train=True, crop_size=args.crop_size, getDmap=args.dnet)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

print('prepare the test dataset')
test_dataset = prepareDataset(graphics, train=False, getDmap=False)
test_loader = DataLoader(test_dataset, batch_size=1)

output_chns = len(args.depth_levels) * 2 + 3
model = Net(args.input_chns, output_chns).cuda()
if args.dnet:
    dnet = DepthEstimatonNet(input_chns=48, output_chns=1).cuda()
    model.load_state_dict(check_point['model_state_dict'])

if args.resume:
    print('load the model parameters for resume training/multitask training from model saved in', check_point['time'])
    model.load_state_dict(check_point['model_state_dict'])
    if args.dnet:
        dnet.load_state_dict(check_point['dnet_state_dict'])

if args.evaluate:
    print('load the model parameters for testing from model saved in', check_point['time'])
    model.load_state_dict(check_point['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

model = nn.DataParallel(model)
if args.dnet:
    dnet = nn.DataParallel(dnet)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
if args.dnet:
    dnet_optimizer = optim.Adam(dnet.parameters(), lr=args.lr)

# print the data retrival information:
# test the data retriving time:
start = time.time()
_ = train_dataset[0] if args.evaluate is None else test_dataset[0]
print('retriving time for data is:', str(time.time() - start) + 's')

def evaluate():
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    psnr = PSNR()
    losses = []
    psnrs = []
    for i, data in enumerate(test_loader):
        torch.cuda.empty_cache()
        # read input shape:
        # data[0]: data input after plane sweep, which directly input to the network
        # data[1]: target x
        # data[0] input shape: N(1), c, h, w where N represents the number of records(One source, one target), default=1
        net_input = data[0]
        source_x = int(data[1])
        target_x = int(data[2])
        reference_img = data[3]

        # read the target image:
        target_img = read_raw_tensorimg(test_dataset.img_folder, target_x)
        target_img = np.transpose(target_img, (2, 0, 1))
        target_img = torch.from_numpy(target_img).float().cuda()
        reference_img = reference_img.float().cuda()

        # target_img shape: (c, h, w)
        net_input = net_input.float().cuda()
        net_output = model(net_input)  # output shape: 1, c, h, w

        # get mpi representation:
        color_imgs, alpha_imgs, reference_x = graphics.get_mpi_singleinput(net_output, source_x, reference_img)

        novel_view = graphics.translation_view_render(color_imgs, alpha_imgs, reference_x, target_x)

        # calculate the mse loss to the result and update the parameters:
        loss = ((novel_view - target_img) ** 2).flatten().sum()
        psnr_value = psnr(novel_view * 255, target_img * 255)
        psnrs.append(psnr_value.cpu().clone().detach().numpy())
        print('psnr:', psnrs[-1])

        # print('loss is: ', loss)
        losses.append(int(loss.cpu().clone().detach().numpy()))
        cv2.imwrite("evaluate/evaluate_sample" + str(i) + ".png",
                    np.transpose((novel_view * 255).cpu().clone().detach().numpy(), (1, 2, 0)))
        print('losses:', losses[-1])

    if args.getmpi:
        graphics.get_mpi(color_imgs, alpha_imgs)

    print('final average loss:', sum(losses) / len(losses))
    print('average PSNR:', sum(psnrs) / len(psnrs))
    return sum(psnrs) / len(psnrs)

if not args.evaluate:
    # training:
    print('total epochs:', args.total_epochs)
    print('total iterations:', len(train_loader) * args.total_epochs)
    print("training===========>")
    iteration_counter = 0
    first_epoch = True

    for epoch in range(0 if args.checkpoint_path is None else check_point['epoch'], args.total_epochs):
        for data in train_loader:

            # clear the grad:
            optimizer.zero_grad()
            if args.dnet:
                dnet_optimizer.zero_grad()
            # read input shape:
            # data[0]: data input after plane sweep, which directly input to the network
            # data[1]: target x
            # data[0] input shape: N(1), c, h, w where N represents the number of records(One source, one target), default=1
            net_input = data[0]
            source_x = data[1]
            target_x = data[2]
            target_imgs = data[3]
            reference_img = data[4]
            if args.dnet:
                depthmap = data[5]
            N = net_input.shape[0]
            h, w = net_input.shape[2], net_input.shape[3]

            # target_img shape: (c, h, w)
            net_input = net_input.float().cuda()
            target_imgs = target_imgs.float().cuda()
            reference_img = reference_img.float().cuda()
            depthmap = depthmap.float().cuda()
            net_output = model(net_input)  # output shape: 1, c, h, w

            # get mpi representation:
            color_imgs, alpha_imgs, reference_x = graphics.get_mpi_multinput(net_output, source_x, reference_img)
            novel_view = torch.zeros((N, 3, h, w)).cuda()

            for i in range(N):
                novel_view[i] = graphics.translation_view_render(color_imgs[i], alpha_imgs[i], int(reference_x[i]), int(target_x[i]))

            # calculate the mse loss to the result and update the parameters:
            loss = ((novel_view - target_imgs) ** 2).flatten().sum() / N
            if args.fvistrain:
                first_visible_img = graphics.get_first_visible_imgs(graphics.get_visible_mpi(color_imgs, alpha_imgs))
                first_visible_loss = ((first_visible_img - reference_img) ** 2).flatten().sum() / N
                loss = loss * 0.7 + first_visible_loss * 0.3

            if args.dnet:
                gray_visible_mpi = graphics.get_visible_mpi(color_imgs, alpha_imgs)
                synthesizedDepthmap = dnet(gray_visible_mpi)
                # calculate the loss of the depth map:
                depth_loss = ((synthesizedDepthmap - depthmap) ** 2).flatten().sum() / N
                loss = loss * 0.7 + depth_loss * 0.3

            loss.backward()
            if (args.dnet and not first_epoch) or (args.dnet and args.resume) or (not args.dnet):
                print('optimize the model')
                optimizer.step()
            if args.dnet:
                dnet_optimizer.step()

            # post processing: save model and sample fig:
            iteration_counter = iteration_counter + 1
            epoch_finished = epoch if not (i == len(train_loader) - 1) else epoch + 1
            print('epoch ', epoch, 'ite ', iteration_counter, 'loss is: ', float(loss.clone().detach().cpu().item()))
            current_checkpoint = {'model_state_dict': model.module.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                            'psnr_max': psnr_max}
            if args.dnet:
                current_checkpoint['dnet_state_dict'] = dnet.module.state_dict()
            torch.save(current_checkpoint, 'checkpoint_{}.pth'.format(args.project_name))

        first_epoch = False
        psnr = evaluate()
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        if psnr > psnr_max:
            current_checkpoint = {'model_state_dict': model.module.state_dict(),
                                  'epoch': epoch,
                                  'args': args,
                                  'time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                  'psnr_max': psnr}
            if args.dnet:
                current_checkpoint['dnet_state_dict'] = dnet.module.state_dict()
            torch.save(current_checkpoint, 'checkpoint_best_{}.pth'.format(args.project_name))
            psnr_max = psnr
else:
    evaluate()