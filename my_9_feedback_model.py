import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
from math import pi,cos,sin
import numpy as np
# import matlab.engine
# import matlab
# import scipy.io as sio
from util import util
import torchvision.transforms as transforms
from PIL import Image
from scipy.misc import imresize

class My9FeedbackModel(BaseModel):
    def name(self):
        return 'My_9_feedback_Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        # parser.set_defaults(dataset_mode='aligned')
        # parser.set_defaults(netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake' ,
                           'Feedback_1', 'Feedback_2','Feedback_3',
                           'Feedback_4','Feedback_5','Feedback_6',
                           'Feedback_7','Feedback_8','Feedback_9' ,
                            'Feedback_all']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A1', 'real_A2', 'real_A3',
                             'real_A4', 'real_A5', 'real_A6',
                             'real_A7', 'real_A8', 'real_A9',
                             'fake_B', 'real_B']

        self.stripe_names = [ 'back_A1', 'back_A2','back_A3',
                              'back_A4', 'back_A5','back_A6',
                              'back_A7','back_A8','back_A9']



        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                          self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1= torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # AtoB = self.opt.direction == 'AtoB'
        #  real_A1   [1,1,512,512]          #  real_b   [1,1,256,256]
        self.real_A1 = input['A1'].to(self.device)  # if AtoB else 'B'
        self.real_A2 = input['A2'].to(self.device)
        self.real_A3 = input['A3'].to(self.device)

        self.real_A4 = input['A4'].to(self.device)  # if AtoB else 'B'
        self.real_A5 = input['A5'].to(self.device)
        self.real_A6 = input['A6'].to(self.device)

        self.real_A7 = input['A7'].to(self.device)  # if AtoB else 'B'
        self.real_A8 = input['A8'].to(self.device)
        self.real_A9 = input['A9'].to(self.device)

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)  # if AtoB else 'A'
        self.image_paths = input['A_paths']  # if AtoB else 'B_paths'

    def forward(self):
        self.fake_B = self.netG(self.real_A)


    def feedback(self):

        w = self.fake_B.shape[2]
        wo = Variable(torch.Tensor(np.array(w / 2)), requires_grad=True)
        k2 = 75.23
        ModFac = 0.8
        aA = Variable(torch.Tensor(np.array(0.5 * ModFac)), requires_grad=True)
        aB = Variable(torch.Tensor(np.array(0.5 * ModFac)), requires_grad=True)
        aC = Variable(torch.Tensor(np.array(0.5 * ModFac)), requires_grad=True)
        mA = Variable(torch.Tensor(np.array(0.5)), requires_grad=True)
        mB = Variable(torch.Tensor(np.array(0.5)), requires_grad=True)
        mC = Variable(torch.Tensor(np.array(0.5)), requires_grad=True)

        x = np.arange(w)
        y = np.arange(w)

        [X, Y] = np.meshgrid(x, y)
        X = Variable(torch.from_numpy(X).float(), requires_grad=True)
        Y = Variable(torch.from_numpy(Y).float(), requires_grad=True)

        # #  illunination phase shifts along the three directions
        p0Ao = Variable(torch.Tensor(np.array(0 * pi / 3)), requires_grad=True)
        p0Ap = Variable(torch.Tensor(np.array(2 * pi / 3)), requires_grad=True)
        p0Am = Variable(torch.Tensor(np.array(4 * pi / 3)), requires_grad=True)
        p0Bo = Variable(torch.Tensor(np.array(0 * pi / 3)), requires_grad=True)
        p0Bp = Variable(torch.Tensor(np.array(2 * pi / 3)), requires_grad=True)
        p0Bm = Variable(torch.Tensor(np.array(4 * pi / 3)), requires_grad=True)
        p0Co = Variable(torch.Tensor(np.array(0 * pi / 3)), requires_grad=True)
        p0Cp = Variable(torch.Tensor(np.array(2 * pi / 3)), requires_grad=True)
        p0Cm = Variable(torch.Tensor(np.array(4 * pi / 3)), requires_grad=True)

        # Illuminating patterns
        thetaA = Variable(torch.Tensor(np.array(0 * pi / 3)), requires_grad=True)
        thetaB = (1 * pi / 3)
        thetaC = (2 * pi / 3)

        # k2a = torch.FloatTensor(np.array([(k2 / w) * cos(thetaA),(k2 / w) * sin(thetaA)]))
        k2a = Variable(torch.FloatTensor(np.array([(k2 / w) * cos(thetaA), (k2 / w) * sin(thetaA)])),
                       requires_grad=True)
        k2b = Variable(torch.FloatTensor(np.array([(k2 / w) * cos(thetaB), (k2 / w) * sin(thetaB)])),
                       requires_grad=True)
        k2c = Variable(torch.FloatTensor(np.array([(k2 / w) * cos(thetaC), (k2 / w) * sin(thetaC)])),
                       requires_grad=True)

        #  illunination phase shifts along the three directions

        # random phase shift errors
        t = torch.rand(9, 1)
        NN = torch.FloatTensor(1 * (0.5 - t) * pi / 18)

        # illunination phase shifts with random errors
        psAo = Variable(p0Ao + NN[0], requires_grad=True)
        psAp = Variable(p0Ap + NN[1], requires_grad=True)
        psAm = Variable(p0Am + NN[2], requires_grad=True)
        psBo = Variable(p0Bo + NN[3], requires_grad=True)
        psBp = Variable(p0Bp + NN[4], requires_grad=True)
        psBm = Variable(p0Bm + NN[5], requires_grad=True)
        psCo = Variable(p0Co + NN[6], requires_grad=True)
        psCp = Variable(p0Cp + NN[7], requires_grad=True)
        psCm = Variable(p0Cm + NN[8], requires_grad=True)
        # r= torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)))

        #  illunination patterns
        sAo = Variable(mA + aA * torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)) + psAo), requires_grad=True)
        sAp = Variable(mA + aA * torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)) + psAp), requires_grad=True)
        sAm = Variable(mA + aA * torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)) + psAm), requires_grad=True)

        sBo = Variable(mB + aB * torch.cos(2 * pi * (k2b[0] * (X - wo) + k2b[1] * (Y - wo)) + psBo), requires_grad=True)
        sBp = Variable(mB + aB * torch.cos(2 * pi * (k2b[0] * (X - wo) + k2b[1] * (Y - wo)) + psBp), requires_grad=True)
        sBm = Variable(mB + aB * torch.cos(2 * pi * (k2b[0] * (X - wo) + k2b[1] * (Y - wo)) + psBm), requires_grad=True)

        sCo = Variable(mC + aC * torch.cos(2 * pi * (k2c[0] * (X - wo) + k2c[1] * (Y - wo)) + psCo), requires_grad=True)
        sCp = Variable(mC + aC * torch.cos(2 * pi * (k2c[0] * (X - wo) + k2c[1] * (Y - wo)) + psCp), requires_grad=True)
        sCm = Variable(mC + aC * torch.cos(2 * pi * (k2c[0] * (X - wo) + k2c[1] * (Y - wo)) + psCm), requires_grad=True)

        # superposed Objects
        self.s1a = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sAo.to(self.device)), requires_grad=True)
        self.s2a = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sAp.to(self.device)), requires_grad=True)
        self.s3a = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sAm.to(self.device)), requires_grad=True)
        self.s1b = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sBo.to(self.device)), requires_grad=True)
        self.s2b = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sBp.to(self.device)), requires_grad=True)
        self.s3b = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sBm.to(self.device)), requires_grad=True)
        self.s1c = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sCo.to(self.device)), requires_grad=True)
        self.s2c = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sCp.to(self.device)), requires_grad=True)
        self.s3c = Variable(torch.Tensor.mul(torch.squeeze(self.fake_B), sCm.to(self.device)), requires_grad=True)


    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B

        #  fake_AB [1,4,256,256]     pred_fake [1,1,30,30]
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
          #  detach() 让梯度不要通过 fake_AB反传到netG中
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def feed_physical(self):

        self.loss_Feedback_1 = self.criterionL1(self.s1a, torch.squeeze(self.real_A1))
        self.loss_Feedback_2 = self.criterionL1(self.s2a, torch.squeeze(self.real_A2))
        self.loss_Feedback_3 = self.criterionL1(self.s3a, torch.squeeze(self.real_A3))
        self.loss_Feedback_4 = self.criterionL1(self.s1b, torch.squeeze(self.real_A4))
        self.loss_Feedback_5 = self.criterionL1(self.s2b, torch.squeeze(self.real_A5))
        self.loss_Feedback_6 = self.criterionL1(self.s3b, torch.squeeze(self.real_A6))
        self.loss_Feedback_7 = self.criterionL1(self.s1c, torch.squeeze(self.real_A7))
        self.loss_Feedback_8 = self.criterionL1(self.s2c, torch.squeeze(self.real_A8))
        self.loss_Feedback_9 = self.criterionL1(self.s3c, torch.squeeze(self.real_A9))

        # self.loss_Feedback_1.backward(retain_graph=True)

        self.loss_Feedback_all = self.loss_Feedback_1 + self.loss_Feedback_2 + self.loss_Feedback_3 + \
                                 self.loss_Feedback_4 + self.loss_Feedback_5 + self.loss_Feedback_6 + \
                                 self.loss_Feedback_7 + self.loss_Feedback_8 + self.loss_Feedback_9
        self.loss_Feedback_all.backward(retain_graph=True)

    def optimize_parameters(self):

        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)    # 更新生成器的时候鉴别器的参数loss不需要更新
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update feedback .
        self.feedback()


        self.feed_physical()
       
        # self.optimizer_G.step()
        # self.optimizer_D.step()







