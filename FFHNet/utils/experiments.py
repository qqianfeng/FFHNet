import collections
### Accuracy
rb1_acc_pos = [0.5, 0.56941, 0.71527, 0.72120, 0.79616, 0.76244, 0.75669, 0.77396]
rb1_acc_neg = [0.5, 0.87966, 0.89288, 0.90329, 0.87451, 0.90322, 0.90379, 0.90419]

## In this test I changed the number of Res Blocks for the Evaluator
#2021-04-30T19_25_36-rb_02
rb2_acc_pos = [0.5, 0.86871, 0.89051, 0.88100, 0.89180, 0.92375, 0.90161, 0.87317]
rb2_acc_neg = [0.5, 0.87591, 0.90722, 0.90823, 0.90739, 0.89342, 0.90853, 0.91654]

#2021-04-28T15_47_17-rb_03
rb3_acc_pos = [0.5, 0.8921, 0.8775, 0.8515, 0.8704, 0.8835, 0.8652, 0.8552]
rb3_acc_neg = [0.5, 0.7106, 0.8887, 0.9311, 0.9351, 0.9152, 0.9265, 0.9165]

rb3_pos = [0.5, 0.856, 0.864, 0.867, 0.895, 0.903, 0.882, 0.88]
rb3_neg = [0.5, 0.885, 0.913, 0.917, 0.909, 0.908, 0.915, 0.9]

#2021-04-29T19_32_54-rb_04
rb4_acc_neg = [0.5, 0.90476, 0.90814, 0.90709, 0.91747, 0.91573, 0.90927, 0.91762]
rb4_acc_pos = [0.5, 0.83255, 0.87308, 0.89200, 0.87158, 0.88228, 0.89313, 0.87442]

#2021-04-30T19_25_14-new_arch
no_res_acc_pos = [0.5, 0.56636, 0.67868, 0.74415, 0.79392, 0.84752, 0.83110, 0.84041]
no_res_acc_neg = [0.5, 0.82974, 0.86432, 0.87737, 0.89768, 0.88262, 0.89336, 0.89266]

### Test on num layers
accs_exps_l = {
    'rb1': (rb1_acc_pos, rb1_acc_neg),
    'rb2': (rb2_acc_pos, rb2_acc_neg),
    'rb3': (rb3_pos, rb3_neg),
    'rb4': (rb4_acc_pos, rb4_acc_neg),
    'nr': (no_res_acc_pos, no_res_acc_neg)
}

# 2021-05-02T19_14_37-rb_03_256
rb3_256_acc_pos = [0.5, 0.87660, 0.87923, 0.89843, 0.86066, 0.87821, 0.90835, 0.88919]
rb3_256_acc_neg = [0.5, 0.88442, 0.90538, 0.89932, 0.91698, 0.91586, 0.90042, 0.91370]

# 2021-04-28T15_47_17-rb_03_512
rb3_512_acc_pos = [0.5, 0.8921, 0.8775, 0.8515, 0.8704, 0.8835, 0.8652, 0.8752]
rb3_512_acc_neg = [0.5, 0.7106, 0.8887, 0.9311, 0.9351, 0.9152, 0.9265, 0.9165]

# 2021-05-01T17_10_15-rb_03_1024
rb3_1024_acc_pos = [0.5, 0.89376, 0.89186, 0.88805, 0.87498, 0.88243, 0.89265, 0.90392]
rb3_1024_acc_neg = [0.5, 0.88213, 0.90191, 0.91326, 0.91596, 0.91471, 0.91208, 0.90791]

# 2021-05-02T19_13_23-rb_03_2048
rb3_2048_acc_pos = [0.5, 0.61458, 0.76215, 0.80535, 0.80854, 0.81942, 0.83249, 0.80173]
rb3_2048_acc_neg = [0.5, 0.83555, 0.84343, 0.88676, 0.89288, 0.89778, 0.89693, 0.90611]

### Test on num neurons
accs_exps_n = collections.OrderedDict()
accs_exps_n['rb3-256'] = (rb3_256_acc_pos, rb3_256_acc_neg)
accs_exps_n['rb3-512'] = (rb3_pos, rb3_neg)
accs_exps_n['rb3-1024'] = (rb3_1024_acc_pos, rb3_1024_acc_neg)
accs_exps_n['rb3-2048'] = (rb3_2048_acc_pos, rb3_2048_acc_neg)

### Dict which houses both tests
exps = {'layers': accs_exps_l, 'neurons': accs_exps_n}

epochs = [0, 5, 10, 15, 20, 25, 30, 35]
if __name__ == '__main__':
    best_acc = 0
    best_arch = ''
    best_epoch = 0
    for arch in accs_exps_n.keys():
        print "Evaluating arch: %s" % arch
        p_list, n_list = accs_exps_n[arch]
        for i, epoch in enumerate(epochs):
            p, n = p_list[i], n_list[i]
            comb_acc = p + 3.5 * n
            print "@epoch: %d, value: %f" % (epoch, comb_acc)
            if comb_acc > best_acc:
                best_acc = comb_acc
                best_arch = arch
                best_epoch = epoch

    print "Overall best arch: %s" % best_arch
    print "@ epoch: %d" % best_epoch
