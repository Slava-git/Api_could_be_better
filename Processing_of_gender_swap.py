import imageio
import requests
import bz2
from PIL import Image
import torch
import torchvision.transforms as transforms
import dlib
from pix2pixHD.data.base_dataset import __scale_width
from pix2pixHD.models.networks import define_G
import pix2pixHD.util.util as util
from aligner import align_face
from api.constants import RESULTS_FOLDER
import os
from api.constants import UPLOAD_FOLDER,THIS_FOLDER

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def download(url, file_name):
    with open(file_name, "wb") as file:
        response = requests.get(url)
        file.write(response.content)

shape_model_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
shape_model_path = 'landmarks.dat'
download(shape_model_url, shape_model_path)
shape_predictor = dlib.shape_predictor(unpack_bz2(shape_model_path))

def get_eval_transform(loadSize=512):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale_width(img,
                                                                      loadSize,
                                                                      Image.BICUBIC)))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

transform = get_eval_transform()

config_G = {
    'input_nc': 3,
    'output_nc': 3,
    'ngf': 64,
    'netG': 'global',
    'n_downsample_global': 4,
    'n_blocks_global': 9,
    'n_local_enhancers': 1,
    'norm': 'instance',
}
FOLDER = os.path.dirname(os.path.abspath(__file__))
weights_path_to_female = os.path.join(FOLDER, 'to_female_net_G.pth')
weights_path_to_male = os.path.join(FOLDER,'to_male_net_G.pth')

def gender_swap():
    img_filename = os.path.join(UPLOAD_FOLDER, "image.jpg")

    aligned_img = align_face(img_filename, shape_predictor)[0]
    img = transform(aligned_img).unsqueeze(0)

    model_to_male = define_G(**config_G)
    pretrained_dict_to_male = torch.load(weights_path_to_male)

    model_to_male.load_state_dict(pretrained_dict_to_male)
    model_to_male.cuda()

    model_to_female = define_G(**config_G)
    pretrained_dict_to_female = torch.load(weights_path_to_female)

    model_to_female.load_state_dict(pretrained_dict_to_female)
    model_to_female.cuda()

    with torch.no_grad():
        out_male = model_to_male(img.cuda())
        out_female = model_to_female(img.cuda())

    out_male = util.tensor2im(out_male.data[0])
    out_female = util.tensor2im(out_female.data[0])

    path_male=os.path.join(THIS_FOLDER,RESULTS_FOLDER,'result_male.jpg')
    path_female = os.path.join(THIS_FOLDER,RESULTS_FOLDER,'result_female.jpg')
    imageio.imsave(path_male, out_male)
    imageio.imsave(path_female, out_female)

