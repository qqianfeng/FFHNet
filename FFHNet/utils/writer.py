import os
import time
import numpy as np
try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None
import torch


class Writer:
    def __init__(self, cfg):
        self.name = cfg.name
        self.cfg = cfg
        self.train_log = os.path.join(cfg.save_dir, 'loss_log.txt')
        self.eval_log = os.path.join(cfg.save_dir, 'eval_log.txt')
        self.testacc_log = os.path.join(cfg.eval_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.confidence_acc = 0
        self.ncorrect = 0

        if cfg.is_train and cfg.to_tensorboard and SummaryWriter is not None:
            self.display = SummaryWriter(
                logdir=self.cfg.save_dir + "/tensorboard")  #comment=cfg.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.cfg.is_train:
            with open(self.train_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

            with open(self.eval_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Eval Loss (%s) ================\n' % now)

        with open(self.testacc_log, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_train_loss(self, epoch, i, loss_dict, t, t_data):
        """ prints train or eval loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.1f, data: %.3f)' \
            % (epoch, i, t, t_data)
        for (loss_type, loss_value) in loss_dict.items():
            message += ' %s: %.5f' % (loss_type, loss_value.item())
        print(message)

        with open(self.train_log, "a") as train_log_file:
            train_log_file.write('%s\n' % message)

    def print_current_eval_loss(self, epoch, loss_dict):
        """ prints eval loss to terminal / file """
        print("=============== Eval loss (%s) ================" % str(epoch))
        message = '(epoch: %d)' % (epoch)
        message_acc = '(epoch: %d)' % (epoch)
        for (loss_type, loss_value) in loss_dict.items():
            if torch.is_tensor(loss_value):
                message += ' %s: %.5f' % (loss_type, loss_value.item())
                if 'acc' in loss_type:
                    message_acc += ' %s: %.5f' % (loss_type, loss_value)
            else:
                message += ' %s: %.5f' % (loss_type, loss_value)
                if 'acc' in loss_type:
                    message_acc += ' %s: %.5f' % (loss_type, loss_value)

        print(message)

        with open(self.eval_log, "a") as eval_log_file:
            eval_log_file.write('%s\n' % message)
        with open(self.testacc_log, "a") as testacc_log_file:
            testacc_log_file.write('%s\n' % message_acc)

    def plot_train_loss(self, loss_dict, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            for (loss_type, loss_value) in loss_dict.items():
                self.display.add_scalar('data/train_loss/' + loss_type, loss_value, iters)

    def plot_eval_loss(self, loss_dict, epoch):
        if self.display:
            for (loss_type, loss_value) in loss_dict.items():
                self.display.add_scalar('data/eval_loss/' + loss_type, loss_value, epoch)

    def plot_model_weights(self, model, epoch):
        if model.train_ffhgenerator and self.display:
            for name, param in model.FFHGenerator.named_parameters():
                self.display.add_histogram('gen_' + name, param.clone().cpu().data.numpy(), epoch)

        if model.train_ffhevaluator and self.display:
            for name, param in model.FFHEvaluator.named_parameters():
                self.display.add_histogram('eva_' + name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        if self.cfg.arch == "evaluator":
            message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
                .format(epoch, acc * 100)
        else:
            message = 'epoch: {}, TEST REC LOSS: [{:.5}]\n' \
                .format(epoch, acc)

        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            if self.cfg.arch == "evaluator":
                self.display.add_scalar('data/test_acc/grasp_prediction', acc, epoch)
            else:
                self.display.add_scalar('data/test_loss/grasp_reconstruction', acc, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.nexamples += nexamples
        self.ncorrect += ncorrect

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()