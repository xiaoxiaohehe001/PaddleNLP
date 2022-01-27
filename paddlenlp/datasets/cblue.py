import collections
import json
import os

import pandas as pd

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder


class CBLUE(DatasetBuilder):
    '''
    The Chinese Biomedical Language Understanding Evaluation (CBLUE) benchmark
    is a collection of natural language understanding tasks including named 
    entity recognition, information extraction, clinical diagnosis normalization
    and single-sentence/sentence-pair classification.
    From https://github.com/CBLUEbenchmark/CBLUE

    CMeEE:
        The Chinese Medical Named Entity Recognition is first released in CHIP20204.
        Given a pre-defined schema, the task is to identify and extract entities 
        from the given sentence and classify them into nine categories: disease, 
        clinical manifestations, drugs, medical equipment, medical procedures, 
        body, medical examinations, microorganisms, and department.

    CMeIE:
        The Chinese Medical Information Extraction is also released in CHIP2020.
        The task is aimed at identifying both entities and relations in a sentence
        following the schema constraints. There are 53 relations defined in the dataset,
        including 10 synonymous sub-relationships and 43 other sub-relationships.

    CHIP-CDN:
        The CHIP Clinical Diagnosis Normalization dataset aims to standardize
        the terms from the final diagnoses of Chinese electronic medical records.

    CHIP-CTC:
        The CHIP Clinical Trial Classification dataset aimed at classifying
        clinical trials eligibility criteria.

    CHIP-STS:
        The CHIP Semantic Textual Similarity dataset consists of question pairs
        related to 5 different diseases and aims to determine sentence similarity.

    KUAKE-QIC:
        The KUAKE Query Intent Classification dataset is used to classify queries
        of search engines into one of 11 medical intent categories, including
        diagnosis, etiology analysis, treatment plan, medical advice, test result
        analysis, disease description, consequence prediction, precautions, intended
        effects, treatment fees, and others.

    KUAKE-QTR:
        The KUAKE Query Title Relevance dataset is used to estimate the 
        relevance of the title of a query document.

    KUAKE-QQR:
        The KUAKE Query-Query Relevance dataset is used to evaluate the 
        relevance of the content expressed in two queries.
    '''

    BUILDER_CONFIGS = {
        'CMeEE': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/CMeEE.zip',
            'md5': '2f21afc5d95918346b673f84eecd06b1',
            'splits': {
                'train': [
                    os.path.join('CMeEE', 'CMeEE_train.json'),
                    '725b34819dd49a0ce028c37e4ad0a73b', ['text']
                ],
                'dev': [
                    os.path.join('CMeEE', 'CMeEE_dev.json'),
                    '42778760dcce7b9ada6e290f7b2a59c2', ['text']
                ],
                'test': [
                    os.path.join('CMeEE', 'CMeEE_test.json'),
                    'c45b3b3d79ca29776e3d9f009b7d6ee5', ['test']
                ]
            },
            'labels': ['dis', 'sym', 'pro', 'equ', 'dru', 'ite', 'bod', 'dep', 'mic']
        },
        'CMeIE': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/CMeIE.zip',
            'md5': '444569dfc31580c8cfa18843d0a1bd59',
            'splits': {
                'train': [
                    os.path.join('CMeIE', 'CMeIE_train.json'),
                    'd27a7d4f0f5326018db66f64ac63780c', ['text']
                ],
                'dev': [
                    os.path.join('CMeIE', 'CMeIE_dev.json'),
                    '54203d1e775a2f07aaea30b61b93ca2f', ['text']
                ],
                'test': [
                    os.path.join('CMeIE', 'CMeIE_test.json'),
                    '8ac74722e9448fdc76132206582b9a06', ['text']
                ]
            },
            'labels': '53_schema.json'
        },
        'CHIP-CDN': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/CHIP-CDN.zip',
            'md5': 'e378d6bfe6740aadfb197ca352db3427',
            'splits': {
                'train': [
                    os.path.join('CHIP-CDN', 'CHIP-CDN_train.json'),
                    '2940ff04e91f52722f10010e5cbc1f18', ['text']
                ],
                'dev': [
                    os.path.join('CHIP-CDN', 'CHIP-CDN_dev.json'),
                    'c718cdd36f913deb11a1a0b46de51015', ['text']
                ],
                'test': [
                    os.path.join('CHIP-CDN', 'CHIP-CDN_test.json'),
                    '8dbe229a23af30bd7c3c5bdcdf156314', ['text']
                ]
            },
            'labels': '国际疾病分类 ICD-10北京临床版v601.xlsx'
        },
        'CHIP-CTC': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/CHIP-CTC.zip',
            'md5': '43d804211d46f9374c18ab13d6984f29',
            'splits': {
                'train': [
                    os.path.join('CHIP-CTC', 'CHIP-CTC_train.json'),
                    '098ac22cafe7446393d941612f906531', ['text']
                ],
                'dev': [
                    os.path.join('CHIP-CTC', 'CHIP-CTC_dev.json'),
                    'b48d52fd686bea286de1a3b123398483', ['text']
                ],
                'test': [
                    os.path.join('CHIP-CTC', 'CHIP-CTC_test.json'),
                    '6a5f0f20f8f85f727d9ef1ea09f939d9', ['text']
                ]
            },
            'labels': 'category.xlsx'
        },
        'CHIP-STS': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/CHIP-STS.zip',
            'md5': '4d4db5ef14336e3179e4e1f3c1cc2621',
            'splits': {
                'train': [
                    os.path.join('CHIP-STS', 'CHIP-STS_train.json'),
                    'c6150e2628f107cf2657feb4ed2ba65b', 
                    ['text1', 'text2']
                ],
                'dev': [
                    os.path.join('CHIP-STS', 'CHIP-STS_dev.json'),
                    '2813ecc0222ef8e4612296776e54639d',
                    ['text1', 'text2']
                ],
                'test': [
                    os.path.join('CHIP-STS', 'CHIP-STS_test.json'),
                    '44394681097024aa922e4e33fa651360',
                    ['text1', 'text2']
                ]
            },
            'labels': ['0', '1']
        },
        'KUAKE-QIC': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/KUAKE-QIC.zip',
            'md5': '7661e3a6b5daf4ee025ba407669788d8',
            'splits': {
                'train': [
                    os.path.join('KUAKE-QIC', 'KUAKE-QIC_train.json'),
                    'fc7e359decfcf7b1316e7833acc97b8a', ['query']
                ],
                'dev': [
                    os.path.join('KUAKE-QIC', 'KUAKE-QIC_dev.json'),
                    '2fd1f4131916239d89b213cc9860c1c6', ['query']
                ],
                'test': [
                    os.path.join('KUAKE-QIC', 'KUAKE-QIC_test.json'),
                    '337dc7f3cdc77b1a21b534ecb3142a6b', ['query']
                ]
            },
            'labels': ['病情诊断', '治疗方案', '病因分析', '指标解读',
                       '就医建议', '疾病表述', '后果表述',
                       '注意事项', '功效作用', '医疗费用', '其他']
        },
        'KUAKE-QTR': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/KUAKE-QTR.zip',
            'md5': 'a59686c2b489ac64ff6f0f029c1df068',
            'splits': {
                'train': [
                    os.path.join('KUAKE-QTR', 'KUAKE-QTR_train.json'),
                    '7197f9ca963f337fc81ce6c8a1c97dc4', ['query', 'title']
                ],
                'dev': [
                    os.path.join('KUAKE-QTR', 'KUAKE-QTR_dev.json'),
                    'e6c480aa46ef2dd04290afe165cdfa9a', ['query', 'title']
                ],
                'test': [
                    os.path.join('KUAKE-QTR', 'KUAKE-QTR_test.json'),
                    '4ccfcf83eef0563b16914d5455d225a5', ['query', 'title']
                ]
            },
            'labels': ['0', '1', '2', '3']
        },
        'KUAKE-QQR': {
            'url': 'https://paddlenlp.bj.bcebos.com/datasets/cblue/KUAKE-QQR.zip',
            'md5': 'b7fdeed0ae56e450d7cf3aa7c0b19e20',
            'splits': {
                'train': [
                    os.path.join('KUAKE-QQR', 'KUAKE-QQR_train.json'),
                    'f667e31610acf3f107369310b78d56a9',
                    ('query1', 'query2')
                ],
                'dev': [
                    os.path.join('KUAKE-QQR', 'KUAKE-QQR_dev.json'),
                    '597354382a806b8168a705584f4f6887',
                    ('query1', 'query2')
                ],
                'test': [
                    os.path.join('KUAKE-QQR', 'KUAKE-QQR_test.json'),
                    '2d257135c6e1651d24a84496dd50c658',
                    ('query1', 'query2')
                ]
            },
            'labels': ['0', '1', '2']
        }
    }


    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, _ = builder_config['splits'][mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(builder_config['url'], default_root,
                              builder_config['md5'])
        return fullname


    def _read(self, filename, split):
        if self.name == 'CMeIE':
            pass
        elif self.name == 'CMeEE':
            pass
        else:
            _, _, input_keys = self.BUILDER_CONFIGS[self.name]['splits'][split]
            with open(filename, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for data in data_list:
                    if data.get('normalized_result', None):
                        data['labels'] = [x.strip('"') for x in data['normalized_result'].split('##')]
                        data.pop('normalized_result')
                    data['text_a'] = data[input_keys[0]]
                    data.pop(input_keys[0])
                    if len(input_keys) > 1:
                        data['text_b'] = data[input_keys[1]]
                        data.pop(input_keys[1])
                    yield data


    def get_labels(self):
        """
        Returns labels of the CBLUE task.
        """
        labels = self.BUILDER_CONFIGS[self.name]['labels']
        if isinstance(labels, str):
            default_root = os.path.join(DATA_HOME, self.__class__.__name__)
            label_dir = os.path.join(default_root, self.name)
        if self.name == 'CHIP-CDN':
            name = [x for x in os.listdir(label_dir) if x.endswith('.xlsx')][0]
            labels = pd.read_excel(os.path.join(label_dir, name), header=None)
            return labels[1].values
        elif self.name == 'CHIP-CTC':
            labels = pd.read_excel(os.path.join(label_dir, labels))
            return labels['Label Name'].values
        elif self.name == 'CMeIE':
            labels = []
            with open(os.path.join(label_dir, labels), 'r') as f:
                for line in f.readlines():
                    labels.append(json.loads(line))
            return labels
        else:
            return self.BUILDER_CONFIGS[self.name]['labels']
