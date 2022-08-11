import os
from FFHNet.config.base_config import BaseConfig


# le originally 5e-4
class TrainConfig(BaseConfig):
    def initialize(self):
        super(TrainConfig, self).initialize()
        self.parser.add_argument('--beta1',
                                 type=float,
                                 default=0.9,
                                 help='First momentum term beta1 of adam')
        self.parser.add_argument('--ds_name',
                                 type=str,
                                 default='train',
                                 help='The name of the dataset.')
        self.parser.add_argument('--eval_freq',
                                 type=int,
                                 default=5,
                                 help='# epochs between evaluating the model')
        self.parser.add_argument('--init_gain',
                                 type=float,
                                 default=0.02,
                                 help='Scaling factor for normal, xavier, orthogonal init.')
        self.parser.add_argument('--is_train',
                                 type=bool,
                                 default=True,
                                 help='Whether training or not.')
        self.parser.add_argument('--kl_coef',
                                 type=float,
                                 default=5e-3,
                                 help='KL divergence coefficient for Coarsenet training.')
        self.parser.add_argument('--lr',
                                 default=1e-4,
                                 type=float,
                                 help='Training learning rate, serving as the initial lr.')
        self.parser.add_argument('--lr_policy',
                                 type=str,
                                 default='plateau',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=40,
                                 help='Total number of epochs for training.')
        self.parser.add_argument(
            '--patience_lr_policy_plateau',
            type=int,
            default=3,
            help=
            'If for #patience times the objective only marginally improved, the learning rate will drop.'
        )
        self.parser.add_argument('--print_freq',
                                 type=int,
                                 default=1,
                                 help='Frequency of printing training results.')
        self.parser.add_argument('--save_epoch_freq',
                                 type=int,
                                 default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_latest_freq',
                                 type=int,
                                 default=1,
                                 help='# of batches between two consecutive saves of the model.')
        self.parser.add_argument('--train_ffhgenerator',
                                 default=False,
                                 type=bool,
                                 help='Whether to train the generator model.')
        self.parser.add_argument('--train_ffhevaluator',
                                 default=True,
                                 type=bool,
                                 help='Whether to train the evaluator model.')
        self.parser.add_argument(
            '--threshold_lr_policy_plateau',
            type=float,
            default=0.01,
            help=
            'Threshold under which a reduction of the loss is considered marginal and will increase the patience.'
        )
        self.parser.add_argument('--weight_decay',
                                 type=float,
                                 default=5e-4,
                                 help='Weight for l2 penalty of optimizer..')
        self.parser.add_argument('--weight_init_type',
                                 type=str,
                                 default='xavier',
                                 help='Use xavier, normal, orthogonal or kaiming weight init?')
        self.is_train = True


if __name__ == '__main__':
    t = TrainConfig().parse()
    print(t)