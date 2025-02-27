from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load', type=str, default=False, help='whether restore or not')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
        parser.add_argument('--num_epochs', type=int, default=1, help='iterations for batch_size samples')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--optimizer', type=str, default='Adam')

        return parser
