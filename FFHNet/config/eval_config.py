from FFHNet.config.base_config import BaseConfig


class EvalConfig(BaseConfig):
    def initialize(self):
        super(EvalConfig, self).initialize()
        self.parser.add_argument('--ds_name',
                                 type=str,
                                 default='eval',
                                 help='The name of the dataset.')
        self.parser.add_argument('--is_train',
                                 type=bool,
                                 default=False,
                                 help='Whether training or not.')
        self.parser.add_argument('--eval_ffhgenerator',
                                 default=False,
                                 type=bool,
                                 help='Whether to evaluate the generator model.')
        self.parser.add_argument('--eval_ffhevaluator',
                                 default=True,
                                 type=bool,
                                 help='Whether to evaluate the evaluator model.')
        self.is_train = False


if __name__ == '__main__':
    cfg = EvalConfig().parse()