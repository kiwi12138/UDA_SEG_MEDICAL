import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import tmpnetworks
class BaseModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_cyc', type=float, default=20.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            
            parser.add_argument('--lambda_fm', type=float, default=1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--noise_std', type=float, default=0, help='stdev of the Gaussian noise')# 0.1
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G_src', 'rec', 'cyc','fm']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A','rec_A', 'fake_B', 'cyc_A']
        visual_names_B = ['real_B','rec_B', 'fake_A', 'cyc_B']


        self.visual_names = visual_names_A + visual_names_B  if self.isTrain else ['real_A','real_B','fake_A','fake_B']# icombine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_E','G_D', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G_E','G_D']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        #self.netG = networks.init_net(tmpnetworks.ResnetGenerator(input_nc=opt.input_nc,ngf=opt.ngf),init_type=opt.init_type, init_gain= opt.init_gain, gpu_ids=self.gpu_ids)
        #self.netG.noise_std=opt.noise_std if self.isTrain else 0
        self.netG_E = networks.init_net(tmpnetworks.ResnetEncoder(input_nc=opt.input_nc,ngf=opt.ngf),init_type='kaiming', init_gain= opt.init_gain, gpu_ids=self.gpu_ids)
        self.netG_D = networks.init_net(tmpnetworks.ResnetDecoder(input_nc=opt.input_nc,ngf=opt.ngf),init_type='kaiming', init_gain= opt.init_gain, gpu_ids=self.gpu_ids)


        if self.isTrain:  # define discriminators
            #self.netD = networks.init_net(nsnetworks.DirectionalDiscriminator(input_nc=opt.input_nc,ndf=opt.ngf),init_type=opt.init_type, init_gain= opt.init_gain, gpu_ids=self.gpu_ids)
            self.netD = networks.init_net(tmpnetworks.DomainDiscriminator(input_nc=opt.input_nc, ndf=opt.ngf, num_classes=2),init_type='normal', init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

            #self.netD_h = networks.init_net(nsnetworks.DirectionalDiscriminator(input_nc=opt.input_nc,ndf=opt.ngf),init_type=opt.init_type, init_gain= opt.init_gain, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionCE=torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_E.parameters(),self.netG_D.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.label_A = torch.zeros(self.real_A.size(0)).long().to(self.device)
        self.label_B = torch.ones(self.real_B.size(0)).long().to(self.device)

    def gaussian(self, in_tensor):
        noisy_image = torch.zeros(list(in_tensor.size())).data.normal_(mean=0, std=self.opt.noise_std).cuda() + in_tensor
        # noisy_tensor = 2 * (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()) - 1
        return noisy_image

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # B mask A images
        if self.isTrain:
            self.real_A_c=self.netG_E(self.real_A,self.label_A)
            self.real_B_c=self.netG_E(self.real_B,self.label_B)

            self.rec_A=self.netG_D(self.real_A_c,self.label_A)
            self.rec_B=self.netG_D(self.real_B_c,self.label_B)

            self.fake_B=self.netG_D(self.real_A_c,self.label_B)
            self.fake_A=self.netG_D(self.real_B_c,self.label_A)

            self.fake_B_c=self.netG_E(self.fake_B,self.label_B)
            self.fake_A_c=self.netG_E(self.fake_A,self.label_A)

            self.cyc_A=self.netG_D(self.fake_B_c,self.label_A)
            self.cyc_B=self.netG_D(self.fake_A_c,self.label_B)
        else:
            #self.netG_E.eval()
            #self.netG_D.eval()
            self.fake_B=self.netG_D(self.netG_E(self.real_A,self.label_A),self.label_B)

            self.fake_A = self.netG_D(self.netG_E(self.real_B, self.label_B), self.label_A)

    def backward_D(self):
        """Calculate GAN loss for discriminator D"""

        loss_D_real = self.criterionGAN(self.netD(torch.cat((self.real_A.detach(), self.real_B.detach()), 0), torch.cat((self.label_A, self.label_B), 0)), True)
        loss_D_fake = self.criterionGAN(self.netD(torch.cat((self.fake_B_pool.query(self.fake_B.detach()), self.fake_A_pool.query(self.fake_A.detach())), 0),torch.cat((self.label_B, self.label_A), 0)), False)
        self.loss_D = (loss_D_real + loss_D_fake)
        if self.opt.gan_mode == 'wgangp':
            raise NotImplementedError
        self.loss_D.backward()

    def backward_G(self):

        loss_G_real = self.criterionGAN(
            self.netD(torch.cat((self.real_A, self.real_B), 0), torch.cat((self.label_A, self.label_B), 0)), False)
        loss_G_fake = self.criterionGAN(
            self.netD(torch.cat((self.fake_B, self.fake_A), 0), torch.cat((self.label_B, self.label_A), 0)), True)
        self.loss_G_src = (loss_G_real + loss_G_fake)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cyc = (self.criterionL1(self.cyc_A, self.real_A) * self.opt.lambda_cyc + self.criterionL1(self.cyc_B,
                                                                                                            self.real_B) * self.opt.lambda_cyc)
        self.loss_rec = (self.criterionL1(self.rec_A, self.real_A) * self.opt.lambda_rec + self.criterionL1(self.rec_B,
                                                                                                            self.real_B) * self.opt.lambda_rec)
        # self.loss_rec_c = (self.criterionL1(self.noise_A_c, self.rec_noise_A_c)+self.criterionL1(self.noise_B_c, self.rec_noise_B_c))/2
        self.loss_fm = (self.criterionL1(self.real_A_c, self.fake_B_c) + self.criterionL1(self.real_B_c,
                                                                                          self.fake_A_c)) * self.opt.lambda_fm
        self.loss_G_same = self.loss_rec
        self.loss_G_cross = self.loss_G_src + self.loss_cyc + self.loss_fm
        self.loss_G_total=self.loss_G_same+self.loss_G_cross
        self.loss_G_total.backward()





    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

