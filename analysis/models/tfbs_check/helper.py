from math import floor
import numpy as np

class Helper:
    PROMOTER_FILE: str = "data/CNNPromoterData/human_non_tata.fa"
    NONPROMOTER_FILE: str = "data/CNNPromoterData/human_nonprom_big.fa"
    _dna_dict: dict
    _lbl_dict: dict
    _variable_dict: dict

    def __init__(self):
        self._dna_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self._lbl_dict = {'Non-Promoter': 0, 'Promoter': 1}
        self._variable_dict = {}

    def get_DNA_dict(self):
        return self._dna_dict

    def get_LABEL_dict(self):
        return self._lbl_dict

    def add_variable(self, name, value):
        try:
            _ = self._variable_dict[name]
            return False
        except KeyError:
            self._variable_dict[name] = value
            return True
    
    def get_variable(self, name):
        try:
            return self._variable_dict[name]
        except KeyError:
            raise Exception("Variable doesn't exist.")

    def update_variable(self, name, value):
        try:
            _ = self._variable_dict[name]
            self._variable_dict[name] = value
            return True
        except KeyError:
            return False

    @staticmethod
    def normalize(x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    @staticmethod
    def softmax(x):
        temp = np.exp(x)
        return (temp / np.sum(temp, axis=0))

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def neg_relu(x, value):
        return min(x, value)

    @staticmethod
    def reciprocal(x):
        temp = 1/x
        return (temp / np.sum(temp, axis=0))

    @staticmethod
    def sum_one(x):
        dif = 1 - np.sum(x, axis=1)
        approx = x + np.expand_dims((dif / 4), axis=1)
        check = np.sum(approx, axis=1)
        if not np.all(check == 1):
            dif = 1 - check
            bad_rows = np.nonzero(check != 1)
            for row in bad_rows[0]:
                if dif[row] > 0:
                    approx[row, np.argmax(approx, axis=1)[row]] += dif[row]
                else:
                    approx[row, np.argmin(approx, axis=1)[row]] += dif[row]
        return approx

    @staticmethod
    def aggregate_weights(filter_idx, filter_weights, position, conv_result_length, weights1=None, weights2=None):
        if weights1 is None:
            weights1 = np.zeros_like(weights2)
        if weights2 is None:
            weights2 = np.zeros_like(weights1)
        return filter_weights * weights1[position + conv_result_length*filter_idx] - filter_weights * weights2[position + conv_result_length*filter_idx]

    @staticmethod
    def conv1d_out_shape(in_length, kernel_size=1, padding=0, stride=1, dilation=1):
        result = floor( ((in_length + (2 * padding) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
        return result

    def get_jaspar_matrices(self, tfbs):
        import coreapi, seqlogo
        helper_class = Helper()
        client = coreapi.Client()
        schema = client.get("http://jaspar.genereg.net/api/v1/docs")
        matrices = {}
        for tfbs_name, collection in tfbs.items():
            action = ["matrix", "list"]
            params = {
                "page_size": '1000',
                "version": 'latest'
            }
            if(tfbs_name is not None):
                params['name'] = tfbs_name
            if(collection is not None):
                params['collection'] = collection
            result_list = client.action(schema, action, params=params)

            action = ["matrix", "read"]
            for res in result_list['results']:
                params = {
                    "matrix_id": res['matrix_id']
                }
                result_read = client.action(schema, action, params=params)
                pfm = np.ndarray((len(result_read['pfm']), len(result_read['pfm']['A'])), dtype=int)
                for key, value in helper_class.get_DNA_dict().items():
                    pfm[value] = np.array(result_read['pfm'][key], dtype=int)
                pm = seqlogo.CompletePm(pfm=pfm)
                matrices[result_read['name']] = pm
        return matrices

    @staticmethod
    def get_filters(matrices, matrix_type):
        max_filter_len = 0
        pms = []
        for m_name, m in matrices.items():
            matrix = getattr(m, matrix_type)
            pm_len = len(matrix)
            if pm_len > max_filter_len:
                max_filter_len = pm_len
            pms.append([m_name, matrix._values.T])
        return max_filter_len, pms