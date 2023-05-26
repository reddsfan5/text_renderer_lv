# coding:utf-8

import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from lmdb_saver import LmdbSaver


class SelectedSampleSaver(object):
    '''
    This class is mainly for coase classification and error sample analysis and based on LmdbSaver!
    '''
    def __init__(self, dataset, labels, preds, probs, idxs, result_name=None, coarse_thresh=0.02):
        self.dataset = dataset
        self.labels = labels
        self.preds = preds
        self.probs = probs
        self.idxs = idxs
        self.result_name = result_name
        self.coarse_thresh = coarse_thresh

    def choose_from_single_lmdb(self, db_loader, idx_bgn, label_set, saver_in=None, errors=None, idxs=None,
                                text_only=False):
        '''
        Choose possible error samples from the current lmdb and save to the coarse the lmdb
        :param db_loader: loader of the current lmdb
        :param idx_bgn: begin and end index of the current lmdb
        :return: None
        '''
        if idxs is None:
            idxs = self.idxs  # [idx_bgn[0]:idx_bgn[1]]
        if saver_in:
            saver = saver_in
        else:
            dst_db_name = db_loader.lmdb_name + '_coarse' + ('' if text_only else '_j2pg')  # append the coarse flag

            saver = LmdbSaver({"lmdb_path": dst_db_name, 'cnt': 0, "cache_capacity": 500})
        if errors is None:
            errors = [0 if T == P else 1 for T, P in zip(self.labels, self.preds)]

        num_selected = 0
        for idx, err in enumerate(errors):
            idx_sample = idxs[idx]
            if isinstance(idx_sample, list):
                idx_sample = idx_sample[0]
            if not err or idx_sample < idx_bgn[0] or idx_sample >= idx_bgn[1]:  #
                continue  # This sample is not in the current lmdb
            # error sample in the current lmdb
            data = db_loader.__getitem__(idx_sample - idx_bgn[0], text_only=text_only, need_trim=False)
            try:
                if data is None or 3 != len(data['image'].shape):
                    print('error sample {} in {}'.format(idx_sample - idx_bgn[0], db_loader.lmdb_name))
                    continue
            except:
                pass
            if idx < len(self.preds):
                data['recoed'] = label_set[self.preds[idx]]
            saver.add(data, is_to_json=True)
            num_selected += 1
        if saver_in is None:
            saver.close()
        print("Total error negative training samples is {}/{}".format(num_selected, np.array(errors).sum()))

    def output_error_sample_to_lmdb(self):
        '''
        output all error sample from the data-set mainly for analysising the error labeled samples
        :param dataset:
        :return:
        '''
        dataset = self.dataset
        err_db_name = dataset.lmdb_loaders.lmdblist[0].lmdb_name + '_err'
        saver = LmdbSaver({"lmdb_path": err_db_name, 'cnt': 0, "cache_capacity": 500})
        loader_list = dataset.lmdb_loaders.lmdblist
        for idx, db_loader in enumerate(loader_list):
            idx_bgn = dataset.lmdb_loaders.set_bgn_idx[idx:]
            self.choose_from_single_lmdb(db_loader, idx_bgn, dataset.lmdb_loaders.label_set, saver)
        saver.close()
        print('end of error samples')

    def classify_negative_sample_by_coarse(self):
        '''
        Choose false positive sample from negative data-set and save to coarse lmdb for next fine training
        :param dataset:
        :return:
        '''
        dataset = self.dataset
        probs = np.array([prob[label] for label, prob in zip(self.labels, self.probs)])
        id_negative = dataset.lmdb_loaders.char_dict.encode('other')
        id_positive = 1 - id_negative
        uni_probs = np.where(np.array(self.labels)==id_negative, 1.0-probs, probs)

        reco_info = list(zip(self.labels, self.idxs, uni_probs))
        # sorted_info = sorted(reco_info, key=lambda x: x[2], reverse=True)  # sorted by the probability
        selected = [x for x in reco_info if x[2] >= self.coarse_thresh]  # moment=0.02, adam=0.4
        labels, idxs, probs = zip(*selected)
        num_pos, num_neg = self.labels.count(id_positive), self.labels.count(id_negative)
        selected_pos, selected_neg = labels.count(id_positive), labels.count(id_negative)
        errors = [True] * len(idxs)
        print('selected positive sample to total {}/{}={}, negative {}/{}={}'
              .format(selected_pos, num_pos, float(selected_pos)/max(1, num_pos),
                      selected_neg, num_neg, float(selected_neg)/max(1, num_neg)))
        loader_list = dataset.lmdb_loaders.lmdblist
        for idx, db_loader in enumerate(loader_list):
            if True: #db_loader.lmdb_name.find('_neg') > 0:
                # errors = [True] * db_loader.num_samples
                # idxs = list(range(db_loader.num_samples))
                idx_bgn = dataset.lmdb_loaders.set_bgn_idx[idx:]
                self.choose_from_single_lmdb(db_loader, idx_bgn, dataset.lmdb_loaders.label_set, None, errors, idxs)
                self.choose_from_single_lmdb(db_loader, idx_bgn, dataset.lmdb_loaders.label_set, None, errors, idxs,
                                             True)

    def output_recognized_result(self):
        if self.result_name is None:
            return
        # if 0 == len(self.labels) and self.result_name is not None:
        dataset, idxs, probs = self.dataset, self.idxs, self.probs

        output = open(self.result_name, 'w', encoding='utf-8')
        label_set = json.dumps({'label_set': dataset.lmdb_loaders.label_set})
        output.write('{"label_set":')
        output.write(label_set)

        # probs = probs.tolist()
        # idxs = idxs.tolist()
        for i, idx in enumerate(idxs):
            if isinstance(idx, list):
                idx = idx[0]
            data_dict = dataset.lmdb_loaders.__getitem__(idx, text_only=True, raw_label=True)
            if 'image' in data_dict:
                data_dict.pop('image')
            if 'id' in data_dict:
                data_dict.pop('id')
            data_dict['preds'] = probs[i]
            jsobj = json.dumps(data_dict)
            output.write(',"{}":'.format(data_dict['trace']))
            output.write(jsobj)

        output.write("}")  # write the end flag
        output.close()

    def compute_confusion(self, title='confuse matrix'):
        classes = set(self.labels)
        cm = confusion_matrix(self.labels, self.preds)
        plt.figure(figsize=(4, 4), dpi=150)
        np.set_printoptions(precision=2)

        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%0.2f" % (c,), color="white" if c > cm.max() / 2 else "black", fontsize=10,
                     va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes)
        plt.yticks(xlocations, classes)
        plt.ylabel('Ground trurh')
        plt.xlabel('Predict')

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', color="gray", linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.05)

        # show confusion matrix
        # plt.savefig(result_path[:-4] + '.png', format='png')
        plt.savefig('temp.png', format='png')
        # plt.show()
        plt.close()

