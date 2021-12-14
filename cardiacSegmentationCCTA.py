import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import sys
import os
import glob
import numpy as np
import SimpleITK as sitk
import visdom
import time
import scipy.ndimage.interpolation as scndy
import argparse

description = '''
This program can be used to train networks for cardiac segmentations 
and to generate segmentations using saved networks.
'''

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', choices=['train', 'test'], type=str, required=True,
                    help="Specify whether to train a network or to test given a saved network.")

# Training-specific arguments
parser.add_argument('--tag', type=str, required=False,
                    help="Only used for mode='train': tag determines the name of the directory in which the networks will be saved.")

parser.add_argument('--train_dir', type=str, required=False,
                    help="Only used for mode='train': directory in which the training images are saved.")

parser.add_argument('--fold', type=str, required=False,
                    help="Only used for mode='train': the training image fold which will be used for training.")

parser.add_argument('--lr', type=float, required=False, default=0.001,
                    help="Sets the (initial) learning rate.")

parser.add_argument('--lr_step_size', type=int, required=False, default=4000,
                    help="Sets the number of iterations after which the learning rate is reduced.")

parser.add_argument('--lr_gamma', type=float, required=False, default=0.3,
                    help="Sets the factor with which the learning rate is reduced every X steps. Must be between 0 and 1")

parser.add_argument('--n_iterations', type=int, required=False, default=10000,
                    help="Sets the total number of iterations for training.")

parser.add_argument('--batch_size', type=int, required=False, default=32,
                    help="Sets the mini-batch size for training.")

# Testing-specific arguments
parser.add_argument('--trained_networks', type=str, nargs="*",
                    help="Specify the trained networks you want to use. If you specify more than one network, these will be used as an ensemble to produce only one output segmentation per input test image.")

parser.add_argument('--test_dir', type=str, required=False,
                    help="Only used for mode='test': directory of the images to be tested.")

# Global variables
parser.add_argument('--rand_seed', type=int, required=False, default=1,
                    help="Sets the random seed for numpy and torch.")

parser.add_argument('--n_class', type=int, required=False, default=6,
                    help="Sets the number of classes to be segmented including background.")

parser.add_argument('--vox_size', type=float, required=False, default=0.8,
                    help="Sets the (isotropic) voxel size the network operates on")

parser.add_argument('--labels_present', dest='labels_present', default=False, action='store_true',
                    help="Only for testing: Set to True if labels are present and Dice scores should be calculated.")

args = parser.parse_args()

np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

NCLASS = args.n_class
VOXSIZE = args.vox_size
LABELPRESENT = args.labels_present
BATCHSIZE = args.batch_size


def load_mhd_to_npy(filename):
    image = sitk.ReadImage(filename)
    spacing = image.GetSpacing()
    return np.swapaxes(sitk.GetArrayFromImage(image), 0, 2), spacing


def loadImageDir3D(imagefiles):
    imagefiles.sort()
    # Images is a list of 3D images
    images = []
    # Labels is a list of 3D masks
    labels = []
    processed = 0

    # Iterate over training images
    for fi in range(0, len(imagefiles)):
        print('Loading ' + str(processed) + '/' + str(len(imagefiles)))
        processed = processed + 1
        f_a = imagefiles[fi]
        reffile_a = f_a.replace('images', 'reference').replace('_image', '_label')

        # If reference file exists
        if os.path.isfile(reffile_a):
            # Load image file
            image, spacing = load_mhd_to_npy(f_a)
            image = image.astype('float32')

            # Resample image to isotropic resolution
            image = scndy.zoom(image, (spacing[0] / VOXSIZE, spacing[1] / VOXSIZE, spacing[2] / VOXSIZE),
                               order=0)

            # Scale image intensities to [0, 1] range
            image[image < -1024.0] = -1024.0
            image[image > 3071.0] = 3071.0
            image = (image + 0.0) / 4096.0

            # Load reference file
            ref, spacing_a = load_mhd_to_npy(reffile_a)

            # Resample reference to isotropic resolution
            ref = scndy.zoom(ref, (spacing[0] / VOXSIZE, spacing[1] / VOXSIZE, spacing[2] / VOXSIZE),
                             order=0)

            images.append(image)
            labels.append(ref)
        else:
            print("Error: Reference file " + reffile_a + " does not exist.")
            sys.exit()

    return images, labels


# Loss function
class DiceLossSimple3D(nn.Module):
    def __init__(self):
        super(DiceLossSimple3D, self).__init__()
        self.dice_loss = dice_loss_simple

    def forward(self, inputs, targets):
        sumdice = 0.0
        for c in range(NCLASS):
            sumdice += self.dice_loss(inputs[:, c, :, :, :].contiguous(), targets[:, c, :, :, :].contiguous())
        return sumdice


# Support function for actual loss function
def dice_loss_simple(input, target):
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def generateBatch3D(images, labels, nsamp=32):
    # Size of each patch 128x128x128 voxels
    sx = 128
    sy = 128
    sz = 128

    # Mini-batch and labels
    batch_im = np.zeros((nsamp, 1, sx, sy, sz))
    batch_la = np.zeros((nsamp, NCLASS, sx, sy, sz))

    for ns in range(nsamp):
        # Randomly select training image
        ind = np.random.randint(0, len(images), 1)
        ind = ind[0]
        imageim = images[ind]
        labelim = labels[ind]

        # Pad image if any direction is smaller than the input patch size
        if imageim.shape[2] < sz:
            offset = sz - imageim.shape[2]
            if offset % 2 == 1:
                offset = offset + 1
            padWidth = (int)(offset / 2)
            imageim = np.pad(imageim, ((0, 0), (0, 0), (padWidth, padWidth)), 'edge')
            labelim = np.pad(labelim, ((0, 0), (0, 0), (padWidth, padWidth)), 'edge')

        if imageim.shape[1] < sy:
            offsety = sy - imageim.shape[1]
            if offsety % 2 == 1:
                offsety = offsety + 1
            padWidthy = (int)(offsety / 2)
            imageim = np.pad(imageim, ((0, 0), (padWidthy, padWidthy), (0, 0)), 'edge')
            labelim = np.pad(labelim, ((0, 0), (padWidthy, padWidthy), (0, 0)), 'edge')

        if imageim.shape[0] < sx:
            offsetx = sx - imageim.shape[0]
            if offsetx % 2 == 1:
                offsetx = offsetx + 1
            padWidthx = (int)(offsetx / 2)
            imageim = np.pad(imageim, ((padWidthx, padWidthx), (0, 0), (0, 0)), 'edge')
            labelim = np.pad(labelim, ((padWidthx, padWidthx), (0, 0), (0, 0)), 'edge')

        offx = np.random.randint(0, imageim.shape[0] - sx + 1)
        offy = np.random.randint(0, imageim.shape[1] - sy + 1)
        offz = np.random.randint(0, imageim.shape[2] - sz + 1)

        imageim = imageim[offx:offx + sx, offy:offy + sy, offz:offz + sz]

        batch_im[ns, 0, :, :, :] = imageim

        labelim = labelim[offx:offx + sx, offy:offy + sy, offz:offz + sz]
        for c in range(NCLASS):
            batch_la[ns, c, :, :, :] = (labelim == c)

    return batch_im, batch_la


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert (padding_type == 'zero')
        p = 1

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator3D(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm3d, n_blocks=9, ngf=32):
        assert (n_blocks >= 0)
        super(ResnetGenerator3D, self).__init__()
        self.ngf = ngf

        model = [nn.Conv3d(1, self.ngf, kernel_size=7, padding=3),
                 norm_layer(self.ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv3d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(self.ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock3D(self.ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=False)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(self.ngf * mult, int(self.ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(self.ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv3d(self.ngf, NCLASS, kernel_size=7, padding=3)]
        model += [nn.Softmax()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output


def test3D(netnames, imdir):
    filenames = glob.glob(imdir + os.path.sep + "*.mhd")
    filenames.sort()
    netdir, netbase = os.path.split(netnames[0])
    imageNo = 0
    if len(filenames) > 0:
        for filename in filenames:
            print("Testing " + filename)
            imageNo += 1
            image = sitk.ReadImage(filename)
            spacing_a = image.GetSpacing()
            origin_a = image.GetOrigin()
            direction_a = image.GetDirection()
            image = sitk.GetArrayFromImage(image)
            image = np.swapaxes(image, 0, 2)
            image = image.astype('float32')

            # Store original shape of the image before downsampling
            orshape = image.shape

            image = scndy.zoom(image, (spacing_a[0] / VOXSIZE, spacing_a[1] / VOXSIZE, spacing_a[2] / VOXSIZE),
                               order=0)

            # Intensity scaling to [0, 1] range
            image[image < -1024.0] = -1024.0
            image[image > 3071.0] = 3071.0
            image = (image + 0.0) / 4096.0

            newshape = image.shape

            wx = int(np.floor(image.shape[0] / 4) * 4)
            wy = int(np.floor(image.shape[1] / 4) * 4)
            wz = int(np.floor(image.shape[2] / 4) * 4)
            if wx > 256:
                wx = 256
            if wy > 256:
                wy = 256
            if wz > 256:
                wz = 256

            batch = np.ones((1, 1, wx, wy, wz), dtype='float32')
            outim_os = np.zeros((NCLASS, image.shape[0], image.shape[1], newshape[2]), dtype='float32')

            # Iterate over networks if an ensemble of several networks is used
            for inet in range(len(netnames)):
                print("    Testing with " + netnames[inet])
                net = ResnetGenerator3D(n_blocks=6, ngf=8)
                net = nn.DataParallel(net)
                net.load_state_dict(torch.load(netnames[inet]))
                net.float()
                net.cuda()
                net.eval()

                if wx == 256 or wy == 256 or wz == 256:
                    nx = int((image.shape[0] + 128) / wx) + 1
                    ny = int((image.shape[1] + 128) / wy) + 1
                    nz = int((image.shape[2] + 128) / wz) + 1
                    for ix in range(nx):
                        for iy in range(ny):
                            for iz in range(nz):
                                batch[0, 0, :, :, :] = image[int(ix * int((image.shape[0] - wx) / (nx - 1))):int(
                                    ix * int((image.shape[0] - wx) / (nx - 1))) + wx,
                                                       int(iy * int((image.shape[1] - wy) / (ny - 1))):int(
                                                           iy * int((image.shape[1] - wy) / (ny - 1))) + wy,
                                                       int(iz * int((image.shape[2] - wz) / (nz - 1))):int(
                                                           iz * int((image.shape[2] - wz) / (nz - 1))) + wz]
                                images_pt = Variable(torch.from_numpy(batch).float().cuda())
                                outputs_pt = net(images_pt).float().cpu().data
                                out = outputs_pt.numpy()
                                outim_os[:, int(ix * int((image.shape[0] - wx) / (nx - 1))):int(
                                    ix * int((image.shape[0] - wx) / (nx - 1))) + wx,
                                int(iy * int((image.shape[1] - wy) / (ny - 1))):int(
                                    iy * int((image.shape[1] - wy) / (ny - 1))) + wy,
                                int(iz * int((image.shape[2] - wz) / (nz - 1))):int(
                                    iz * int((image.shape[2] - wz) / (nz - 1))) + wz] = outim_os[:, int(
                                    ix * int((image.shape[0] - wx) / (nx - 1))):int(
                                    ix * int((image.shape[0] - wx) / (nx - 1))) + wx, int(
                                    iy * int((image.shape[1] - wy) / (ny - 1))):int(
                                    iy * int((image.shape[1] - wy) / (ny - 1))) + wy, int(
                                    iz * int((image.shape[2] - wz) / (nz - 1))):int(
                                    iz * int((image.shape[2] - wz) / (nz - 1))) + wz] + np.squeeze(out[0, :, :, :, :])

                else:
                    # Process overlapping patches and add up probabilities
                    batch[0, 0, :, :, :] = image[:wx, :wy, :wz]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, :wx, :wy, :wz] = outim_os[:, :wx, :wy, :wz] + np.squeeze(out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[image.shape[0] - wx:image.shape[0], :wy, :wz]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, image.shape[0] - wx:image.shape[0], :wy, :wz] = outim_os[:,
                                                                                image.shape[0] - wx:image.shape[0],
                                                                                :wy, :wz] + np.squeeze(
                        out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[:wx, image.shape[1] - wy:image.shape[1], :wz]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, :wx, image.shape[1] - wy:image.shape[1], :wz] = outim_os[:, :wx,
                                                                                image.shape[1] - wy:image.shape[1],
                                                                                :wz] + np.squeeze(
                        out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[:wx, :wy, image.shape[2] - wz:image.shape[2]]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, :wx, :wy, image.shape[2] - wz:image.shape[2]] = outim_os[:, :wx, :wy,
                                                                                image.shape[2] - wz:image.shape[
                                                                                    2]] + np.squeeze(
                        out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[image.shape[0] - wx:image.shape[0],
                                           image.shape[1] - wy:image.shape[1],
                                           :wz]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, image.shape[0] - wx:image.shape[0], image.shape[1] - wy:image.shape[1],
                    :wz] = outim_os[
                           :,
                           image.shape[
                               0] - wx:
                           image.shape[
                               0],
                           image.shape[
                               1] - wy:
                           image.shape[
                               1],
                           :wz] + np.squeeze(
                        out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[image.shape[0] - wx:image.shape[0], :wy,
                                           image.shape[2] - wz:image.shape[2]]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, image.shape[0] - wx:image.shape[0], :wy,
                    image.shape[2] - wz:image.shape[2]] = outim_os[
                                                          :,
                                                          image.shape[
                                                              0] - wx:
                                                          image.shape[
                                                              0],
                                                          :wy,
                                                          image.shape[
                                                              2] - wz:
                                                          image.shape[
                                                              2]] + np.squeeze(
                        out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[:wx, image.shape[1] - wy:image.shape[1],
                                           image.shape[2] - wz:image.shape[2]]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, :wx, image.shape[1] - wy:image.shape[1],
                    image.shape[2] - wz:image.shape[2]] = outim_os[
                                                          :, :wx,
                                                          image.shape[
                                                              1] - wy:
                                                          image.shape[
                                                              1],
                                                          image.shape[
                                                              2] - wz:
                                                          image.shape[
                                                              2]] + np.squeeze(
                        out[0, :, :, :, :])

                    batch[0, 0, :, :, :] = image[image.shape[0] - wx:image.shape[0],
                                           image.shape[1] - wy:image.shape[1],
                                           image.shape[2] - wz:image.shape[2]]
                    images_pt = Variable(torch.from_numpy(batch).float().cuda())
                    outputs_pt = net(images_pt).float().cpu().data
                    out = outputs_pt.numpy()
                    outim_os[:, image.shape[0] - wx:image.shape[0], image.shape[1] - wy:image.shape[1],
                    image.shape[2] - wz:image.shape[2]] = outim_os[:, image.shape[0] - wx:image.shape[0],
                                                          image.shape[1] - wy:image.shape[1],
                                                          image.shape[2] - wz:image.shape[2]] + np.squeeze(
                        out[0, :, :, :, :])

            outim_os_or = np.zeros((NCLASS, orshape[0], orshape[1], orshape[2]))
            for k in range(NCLASS):
                # Reshape to original shape
                outim_os_or[k, :, :, :] = scndy.zoom(outim_os[k, :, :, :], (
                    orshape[0] / newshape[0], orshape[1] / newshape[1], orshape[2] / newshape[2]),
                                                     order=0)

            outnamet = netdir + os.path.sep + os.path.split(filename)[-1]

            outim = np.zeros((outim_os_or.shape[1], outim_os_or.shape[2], outim_os_or.shape[3]))
            for z in range(outim_os_or.shape[3]):
                # Calculate output segmentation mask from class probabilities
                outim[:, :, z] = np.argmax(np.squeeze(outim_os_or[:, :, :, z]), axis=0)

            outimt = np.copy(outim)
            outimt = np.swapaxes(outimt, 0, 2)
            outimt = sitk.GetImageFromArray(outimt)
            outimt.SetSpacing(spacing_a)
            outimt.SetOrigin(origin_a)
            outimt.SetDirection(direction_a)
            # Write output segmentation in same directory where the used network is stored
            sitk.WriteImage(outimt, outnamet.replace('.mhd', '_{}.mhd'.format('9')), True)


def train3D(tag, task, fold, lr, lr_step_size, lr_gamma, n_iterations, expdir):
    print('Experiment tag ' + tag)
    net = ResnetGenerator3D(n_blocks=6, ngf=8)
    net.cuda()
    net.float()
    net = nn.DataParallel(net)

    criterion = DiceLossSimple3D()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Make sure the training images and reference masks are stored at the correct relative path with respect to the python file
    traindir = os.path.dirname(
        os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold_' + str(
        fold) + os.path.sep + 'train' + os.path.sep + 'images'
    print("traindir " + traindir)
    filetype = r'*.mhd'
    trainimages, trainlabels = loadImageDir3D(glob.glob(traindir + os.path.sep + filetype))

    startIt = -1
    num_epochs = n_iterations
    errors = np.empty(startIt + int(num_epochs) + 1)
    errors[:startIt] = 0
    testes1 = np.empty(startIt + int(num_epochs) + 1)
    testes1[:startIt] = 0

    # visdom can be used for online visualization
    viz = visdom.Visdom(env=tag, port=8099)
    itInt = 50

    # Make sure the validation images and reference masks are stored at the correct relative path with respect to the python file
    valdir = os.path.dirname(
        os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold_' + str(
        fold) + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels = loadImageDir3D(glob.glob(valdir + os.path.sep + filetype))

    # A central 128x128x128 voxel patch of one validation image is used for visualization and loss tracking in this case
    valim = valimages[0]
    valims = valim[int(valim.shape[0] / 2) - 64:int(valim.shape[0] / 2) + 64,
             int(valim.shape[1] / 2) - 64:int(valim.shape[1] / 2) + 64,
             int(valim.shape[2] / 2) - 64:int(valim.shape[2] / 2) + 64]
    valimp = valims
    padwidthVal = 0

    if valimp.shape[2] < 128:
        padwidthVal = 128 - valimp.shape[2]
        valimp = np.pad(valimp, ((0, 0), (0, 0), (0, padwidthVal)), 'constant', constant_values=0)
    valimb = np.zeros((1, 1, valimp.shape[0], valimp.shape[1], valimp.shape[2]), dtype='float32')
    valimb[0, 0, :, :, :] = valimp

    valla = vallabels[0]
    valla = valla[int(valim.shape[0] / 2) - 64:int(valim.shape[0] / 2) + 64,
            int(valim.shape[1] / 2) - 64:int(valim.shape[1] / 2) + 64,
            int(valim.shape[2] / 2) - 64:int(valim.shape[2] / 2) + 64]
    if valla.shape[2] < 128:
        valla = np.pad(valla, ((0, 0), (0, 0), (0, padwidthVal)), 'constant', constant_values=0)
    vallab = np.zeros((1, NCLASS, valla.shape[0], valla.shape[1], valla.shape[2]), dtype='float32')
    for cl in range(NCLASS):
        vallab[0, cl, :, :, :] = valla == (cl)

    bs = BATCHSIZE
    # Training loop
    for it in range(num_epochs):
        start = time.time()
        if it > -1:
            optimizer.zero_grad()
            images, labels = generateBatch3D(trainimages, trainlabels, nsamp=bs)
            images, labels = Variable(torch.from_numpy(images).float().cuda()), Variable(
                torch.from_numpy(labels).float().cuda())
            outputstrain = net(images)
            loss = criterion(outputstrain, labels)
            print('Dice loss {}'.format(loss.item()))
            loss.backward()
            scheduler.step()
            optimizer.step()
            errors[it] = loss.item()
        # Validation loss tracking and visualization every itInt iterations
        if it % itInt == 0 and it > 0:
            images_pt = Variable(torch.from_numpy(valimb).float().cuda())
            net.eval()
            outputs_pt = net(images_pt)
            labels_pt = Variable(torch.from_numpy(vallab).float().cuda())
            valloss = criterion(outputs_pt, labels_pt)
            testes1[it:it + itInt] = valloss.item()

            net.train()
            outputs = outputs_pt.float().cpu().data.numpy()

            # Visualization with visdom, not mandatory
            viz.image(np.clip(scndy.zoom(
                (np.rot90(np.flipud(np.squeeze(valims[:, :, 63])), k=3, axes=(0, 1)) + 0.25) * 255,
                1.0), 0.0, 255.0),
                win=1,
                opts=dict(title="Validation image"))

            viz.image(np.clip(
                scndy.zoom((np.rot90(np.flipud(np.squeeze(valla[:, :, 63])), k=3, axes=(0, 1)) + 0.25) * 35, 1.0),
                0.0, 255.0),
                win=2,
                opts=dict(title="Validation labels"))

            for cl in range(NCLASS):
                viz.image(scndy.zoom(
                    np.clip(np.rot90(np.flipud(np.squeeze(outputs[0, cl, :, :, 63])), k=3, axes=(0, 1)) * (255), 0,
                            255), 1.0),
                    win=3 + cl, opts=dict(title="Test class {}".format(cl)))

            vizX = np.zeros((it, 2))
            vizX[:, 0] = range(it)
            vizX[:, 1] = range(it)
            vizY = np.zeros((it, 2))
            vizY[:, 0] = errors[:it]
            vizY[:, 1] = testes1[:it]
            vizY[np.isnan(vizY)] = 0.0
            viz.line(Y=vizY, X=vizX, win='viswin1')

            netnameout = expdir + os.path.sep + str(it) + '.pt'
            torch.save(net.state_dict(), netnameout)
            np.savetxt(expdir + os.path.sep + 'trainloss.txt', errors[:it])
            np.savetxt(expdir + os.path.sep + 'valloss.txt', testes1[:it])
        print('Iteration {} took {} s'.format(it, time.time() - start))


if __name__ == "__main__":
    if args.mode == 'train':
        print("Going to train")
        expdir = os.path.dirname(
            os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + args.tag
        if not os.path.exists(expdir):
            os.makedirs(expdir)
        train3D(args.tag, args.train_dir, args.fold, args.lr, args.lr_step_size, args.lr_gamma, args.n_iterations,
                expdir)
    if args.mode == 'test':
        print("Going to test")
        netdir, netbase = os.path.split(args.trained_networks[0])
        test3D(args.trained_networks, args.test_dir)
