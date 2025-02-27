import argparse

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--dataroot', default='dataset/DRIVE/test/', help='path to data')
        parser.add_argument('--dataroot_test', default='dataset/DRIVE/test/', help='path to data')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        # CF:CF Loss; CE_DC:CE and DC; CUR:CUR Loss; VE
        parser.add_argument('--loss_name', type=str, default='VE', help='The loss name.')
        parser.add_argument('--train_ids',type=list,default=[0,2],help='train id number')
        parser.add_argument('--val_ids',type=list,default=[0,2],help='val id number')
        parser.add_argument('--test_ids',type=list,default=[0,2],help='test id number')
        parser.add_argument('--modality_filename', type=list, default=['av_images', 'av_labels'], help='dataset filename, last name is label filename')
        parser.add_argument('--data_size', type=list, default=[576,576], help='input data size separated with comma')
        parser.add_argument('--in_channels', type=int, default=3, help='input channels')
        parser.add_argument('--channels', type=int, default=128, help='channels')
        parser.add_argument('--saveroot', default='logs', help='path to save results')
        parser.add_argument('--n_classes', type=int, default=3, help='final class number for classification')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



