# System libs
import os
import argparse
# Numerical libs
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from scipy.io import loadmat
from scipy.misc import imread, imresize, imsave
from scipy.ndimage import zoom
# Our libs
from models import ModelBuilder
from utils import colorEncode
# vis
from tensorboardX import SummaryWriter

# forward func for testing
def forward_test_multiscale(nets, img, args):
    (net_encoder, net_decoder) = nets

    pred = torch.zeros(1, args.num_class, img.size(2), img.size(3))
    if torch.cuda.is_available():
        pred = Variable(pred, volatile=True).cuda()
    else:
        pred = Variable(pred, volatile=True)

    for scale in args.scales:
        img_scale = zoom(img.numpy(),
                         (1., 1., scale, scale),
                         order=1,
                         prefilter=False,
                         mode='nearest')

        # feed input data
        if torch.cuda.is_available():
            input_img = Variable(torch.from_numpy(img_scale),
                             volatile=True).cuda()
        else:
            input_img = Variable(torch.from_numpy(img_scale),
                             volatile=True)

        # forward
        pred_scale = net_decoder(net_encoder(input_img),
                                 segSize=(img.size(2), img.size(3)))

        # average the probability
        pred = pred + pred_scale / len(args.scales)

    return pred


def visualize_test_result(img, pred, args):
    colors = loadmat('data/color150.mat')['colors']
    # recover image
    img = img[0]
    pred = pred.data.cpu()[0]
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

    # prediction
    pred_ = np.argmax(pred.numpy(), axis=0)
    #---------added----------
    pplornot = np.where(pred_==12,1,0)
    np.savetxt(args.test_img + '.txt', pplornot, delimiter=',', fmt='%u')
    #-----------------
    pred_color = colorEncode(pred_, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1).astype(np.uint8)
    paths = os.path.join('/tmpdata/result', os.path.basename(args.test_img) + '.png')
    imsave(paths, im_vis)
    print(paths)
    bucket_name = "ubiquity-kube-mlengine-pytorch_trial/ade20k"
    upload_to_buckets('result/'+os.path.basename(args.test_img) + '.png', bucket_name)


def test(nets, args):
    # switch to eval mode
    for net in nets:
        net.eval()

    # loading image, resize, convert to tensor
    img = imread(args.test_img, mode='RGB')

    h, w = img.shape[0], img.shape[1]
    s = 1. * args.imgSize / min(h, w)
    img = imresize(img, s)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img = img_transform(img)
    img = img.view(1, img.size(0), img.size(1), img.size(2))

    # forward pass
    pred = forward_test_multiscale(nets, img, args)

    # visualization
    visualize_test_result(img, pred, args)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def cloud_local_file_exchagne(start_location, destination_folder, file_name):
    # pull or push data between buckets and local directory
    # return the path of destination
    # will create local folder to save stuff
    # bucket has to exist for pushing stuff

    # start_location: string, e.g. gs://bucket_name/sub_folder, without file names
    # destination folder: string, e.g. /full/path/on/local/machine, without file names
    from subprocess import call
    if start_location[:5] == "gs://":
        # downloading data, make sure destination folder exists
        call(['mkdir', '-p', destination_folder])

    if file_name == "":
        # copying full folder
        start_location = start_location+'/*'
        call(['gsutil','-m','cp','-r', start_location, destination_folder])
        return destination_folder

    full_start_path = start_location + '/' + file_name
    full_des_path = destination_folder + '/' + file_name

    call(['gsutil', 'cp', full_start_path, full_des_path])
    return full_des_path

def upload_to_buckets(local_folder, destination_bucket_name):
    # push data to buckets
    # local_folder is presumed to be under /tmpdata/
    # all files under local_folder will be copied to the bucket, path preserved
    from subprocess import call

    destination_bucket_url = 'gs://' + destination_bucket_name + local_folder
    call(['gsutil','cp', '-r', '/tmpdata/'+local_folder, destination_bucket_url])


def main(args):
    # download data from buckets
    #download_from_buckets(args.test_img)
    bucket_name = "ubiquity-kube-mlengine-pytorch_trial"
    source_blob_name = "ade20k/davidgarrett.jpg"
    destination_file_name = "/davidgarrett.jpg"
    download_blob(bucket_name, source_blob_name, destination_file_name)
    args.test_img = destination_file_name


    bucket_url = args.weights_encoder
    print(bucket_url, type(bucket_url))
    destination_folder = 'model_path_en'
    args.weights_encoder = download_from_buckets(bucket_url, destination_folder)
    bucket_url = args.weights_decoder
    print(bucket_url, type(bucket_url))
    destination_folder = 'model_path_de'
    args.weights_decoder = download_from_buckets(bucket_url, destination_folder)
    print('gsutil works')

    # Network Builders
    writer = SummaryWriter()

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(arch=args.arch_decoder,
                                        fc_dim=args.fc_dim,
                                        segSize=args.segSize,
                                        weights=args.weights_decoder,
                                        use_softmax=True)

    nets = (net_encoder, net_decoder)
    if torch.cuda.is_available():
        for net in nets:
            net.cuda()

    # single pass
    test(nets, args)

    print('Done! Output is saved in {}'.format(args.result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--test_img', required=True)
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_best.pth',
                        help="which snapshot to load")
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='c1_bilinear',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=512, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--imgSize', default=384, type=int,
                        help='resize input image')
    parser.add_argument('--segSize', default=-1, type=int,
                        help='output image size, -1 = keep original')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # scales for evaluation
    args.scales = (0.5, 0.75, 1, 1.25, 1.5)

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    main(args)
