import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pudb


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        ''' Parses opts and initializes the relevant networks. '''
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # self.opt = opt

        # load and define networks according to opts
        self.netG = networks.define_G(
                opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.product_cat:
                self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(
                    use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                    self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized ----------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # run this after setting inputs
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        # no backprop on gradients
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def validation(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)


            if self.opt.L2:
                self.val_loss_G_L = self.criterionL2(self.fake_B, self.real_B)
            else:
                self.val_loss_G_L = self.criterionL1( self.fake_B, self.real_B)
            #print "Val L1 loss: ", self.val_loss_G_L1.item()

        return self.val_loss_G_L.item()

    def get_image_paths(self):
        return self.image_paths

    def cat_AB(self, A, B):
        '''
        Product each contour in A with B
        :param A:
        :param B:
        :return: A*B
        '''
        # C = A.clone()
        # C = A * B
        return A*B

    def backward_D(self):
        # first stop backprop to the generator by detaching fake_B
        # add real_A, fake_B to query. Concatenates on axis=1
        # TODO: Why do we detach?
        if self.opt.product_cat:
            fake_AB = self.cat_AB( self.real_A, self.fake_B)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)

            # fake_AB = self.fake_AB_pool.query(
            #     torch.cat((self.real_A, self.fake_B), 1).data)

        fake_AB = self.fake_AB_pool.query( fake_AB.data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # real
        if self.opt.product_cat:
            real_AB = self.cat_AB( self.real_A, self.real_B)
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_D_semi(self):
        # first stop backprop to the generator by detaching fake_B
        # add real_A, fake_B to query. Concatenates on axis=1
        # TODO: Why do we detach?
        if self.opt.product_cat:
            fake_AB = self.cat_AB( self.real_A, self.fake_B)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)

            # fake_AB = self.fake_AB_pool.query(
            #     torch.cat((self.real_A, self.fake_B), 1).data)

        fake_AB = self.fake_AB_pool.query( fake_AB.data)
        # fake_AB = self.fake_AB_pool.query(
        #         torch.cat((self.real_A, self.fake_B), 1).data)

        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake_semi = self.criterionGAN(pred_fake, False)

        lambdaD_semi    = 1
        self.loss_D_semi = lambdaD_semi * self.loss_D_fake_semi

        self.loss_D_semi.backward()

    def backward_G_semi(self):
        # first G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        if self.opt.product_cat:
            fake_AB = self.cat_AB( self.real_A, self.fake_B)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_fake = self.netD(fake_AB)
        #seems similar to backwardD loss but the D label is flip to True
        self.loss_G_GAN_semi = self.criterionGAN(pred_fake, True)

        # lambdaG_semi = .1
        lambdaG_semi = .01
        self.loss_G_semi = lambdaG_semi * self.loss_G_GAN_semi

        self.loss_G_semi.backward()

    def backward_G(self):
        # first G(A) should fake the discriminator
        if self.opt.product_cat:
            fake_AB = self.cat_AB( self.real_A, self.fake_B)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        if self.opt.noGANloss:
            if self.opt.L2:
                # second G(A) = B
                self.loss_G_L1 = self.criterionL2(self.fake_B,
                                                  self.real_B) # * self.opt.lambda_A
            else:
                self.loss_G_L1 = self.criterionL1( self.fake_B, self.real_B)

            self.loss_G = self.loss_G_L1
        else: #original loss
            self.loss_G_L1 = self.criterionL1(self.fake_B,
                                              self.real_B) * self.opt.lambda_A
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters_semi(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D_semi()
        # self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G_semi()
        self.optimizer_G.step()

    def get_current_errors(self):
        # return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]), ('G_L1', self.loss_G_L1.data[0]), ('D_real', self.loss_D_real.data[0]), ('D_fake', self.loss_D_fake.data[0])])
        return OrderedDict(
            [('G_GAN', self.loss_G_GAN.item()), ('G_L1', self.loss_G_L1.item()), ('D_real', self.loss_D_real.item()),
             ('D_fake', self.loss_D_fake.item()),
             ])

    def get_current_errors_semi(self):
        # return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]), ('G_L1', self.loss_G_L1.data[0]), ('D_real', self.loss_D_real.data[0]), ('D_fake', self.loss_D_fake.data[0])])
        return OrderedDict(
            [('G_GAN', self.loss_G_GAN.item()), ('G_L1', self.loss_G_L1.item()), ('D_real', self.loss_D_real.item()),
             ('D_fake', self.loss_D_fake.item()),
             ('G_GAN_semi', self.loss_G_semi.item()) ,
             ('D_fake_semi', self.loss_D_semi.item())
             ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
