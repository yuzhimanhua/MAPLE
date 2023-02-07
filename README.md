# The Effect of Metadata on Scientific Literature Tagging: A Cross-Field Cross-Model Study
**NOTE: We are still finalizing our paper and will add three more datasets: Biology-MeSH, Chemistry-MeSH, and Medicine-MeSH to this repository. More details coming soon!**

This repository contains the datasets and source code used in our paper [The Effect of Metadata on Scientific Literature Tagging: A Cross-Field Cross-Model Study]().

## Links

- [Datasets](#datasets)
- [Running Parabel](#running-parabel)
- [Running Transformer](#running-transformer)
- [Running OAG-BERT](#running-oag-bert)
- [References](#references)

## Datasets
The MAPLE benchmark constructed by us contains 20 datasets across 19 fields for scientific literature tagging. You can download the datasets from [**HERE**](https://gofile.io/d/vMcdjW). Once you unzip the downloaded file, you can see a folder ```MAPLE/```. Please put the folder under the main directory ```./``` of this code repository.

There are 20 folders under ```MAPLE/```, corresponding to the 20 datasets mentioned in the submission. Their statistics are as follows:

| Folder                       | Field                         | #Papers  | #Labels  | #Venues | #Authors  | #References   |
| ---------------------------- | ----------------------------- | -------- | -------- | ------- | --------- | ------------- |
| ```Art```                    | Art                           | 58,373   | 1,990    | 98      | 54,802    | 115,343       |
| ```Philosophy```             | Philosophy                    | 59,296   | 3,758    | 98      | 36,619    | 198,010       |
| ```Geography```              | Geography                     | 73,883   | 3,285    | 98      | 157,423   | 884,632       |
| ```Business```               | Business                      | 84,858   | 2,392    | 97      | 100,525   | 685,034       |
| ```Sociology```              | Sociology                     | 90,208   | 1,935    | 98      | 85,793    | 842,561       |
| ```History```                | History                       | 113,147  | 2,689    | 99      | 84,529    | 284,739       |
| ```Political_Science```      | Political Science             | 115,291  | 4,990    | 98      | 93,393    | 480,136       |
| ```Environmental_Science```  | Environmental Science         | 123,945  | 694      | 100     | 265,728   | 1,217,268     |
| ```Economics```              | Economics                     | 178,670  | 5,205    | 97      | 135,247   | 1,042,253     |
| ```CSRankings```             | Computer Science (Conference) | 263,393  | 13,613   | 75      | 331,582   | 1,084,440     |
| ```Engineering```            | Engineering                   | 270,006  | 10,683   | 100     | 430,046   | 1,867,276     |
| ```Psychology```             | Psychology                    | 372,954  | 7,641    | 100     | 460,123   | 2,313,701     |
| ```Computer_Science```       | Computer Science (Journal)    | 410,603  | 15,540   | 96      | 634,506   | 2,751,996     |
| ```Geology```                | Geology                       | 431,834  | 7,883    | 100     | 471,216   | 1,753,762     |
| ```Mathematics```            | Mathematics                   | 490,551  | 14,271   | 98      | 404,066   | 2,150,584     |
| ```Materials_Science```      | Materials Science             | 1,337,731 | 6,802    | 99      | 1,904,549  | 5,457,773     |
| ```Physics```                | Physics                       | 1,369,983 | 16,664   | 91      | 1,392,070  | 3,641,761     |
| ```Biology```                | Biology                       | 1,588,778 | 64,267   | 100     | 2,730,547  | 7,086,131     |
| ```Chemistry```              | Chemistry                     | 1,849,956 | 35,538   | 100     | 2,721,253  | 8,637,438     |
| ```Medicine```               | Medicine                      | 2,646,105 | 36,619   | 100     | 4,345,385  | 7,405,779     |


## Running Parabel
The code of Parabel is written in C++. It is adapted from [**the original implementation**](http://manikvarma.org/code/Parabel/download.html) by Prabhu et al. You need to run the following script.
```
cd ./Parabel/
./run.sh
```
P@_k_ and NDCG@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./Parabel/scores.txt```. The prediction results can be found in ```./Parabel/Sandbox/Results/{dataset}/score_mat.txt```.

## Running Transformer
The code of Transformer is written in Python 3.6. It is adapted from [**the original implementation**](https://github.com/XunGuangxu/CorNet) by Xun et al. You need to first install the dependencies like this:
```
cd ./Transformer/
pip3 install -r requirements.txt
```
Then, you need to download [**the pre-trained GloVe embeddings**](https://gofile.io/d/S1IJbe). Once you unzip the downloaded file, please put it (i.e., the ```data/``` folder) under ```./Transformer/```. Then, you can run the code (GPUs are strongly recommended).
```
./run.sh
```
P@_k_ and NDCG@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./Transformer/scores.txt```. The prediction results can be found in ```./Transformer/predictions.txt```.

## Running OAG-BERT
The code of OAG-BERT is written in Python 3.7. It is adapted from [**the original implementation**](https://github.com/THUDM/OAG-BERT) by Liu et al. You need to first install **PyTorch >= 1.7.1**, and then the [**CogDL**](https://github.com/THUDM/cogdl) package.
```
pip3 install cogdl
```
Then, you can run the code (GPUs are strongly recommended).
```
cd ./OAGBERT/
./run.sh
```
P@_k_ and NDCG@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./OAGBERT/Parabel/scores.txt```. The prediction results can be found in ```./OAGBERT/Parabel/Sandbox/Results/{dataset}/score_mat.txt```.

## References
```
@inproceedings{prabhu2018parabel,
  title={Parabel: Partitioned label trees for extreme classification with application to dynamic search advertising},
  author={Prabhu, Yashoteja and Kag, Anil and Harsola, Shrutendra and Agrawal, Rahul and Varma, Manik},
  booktitle={WWW'18},
  pages={993--1002},
  year={2018}
}

@inproceedings{xun2020correlation,
  title={Correlation networks for extreme multi-label text classification},
  author={Xun, Guangxu and Jha, Kishlay and Sun, Jianhui and Zhang, Aidong},
  booktitle={KDD'20},
  pages={1074--1082},
  year={2020}
}

@inproceedings{liu2022oag,
  title={Oag-bert: Towards a unified backbone language model for academic knowledge services},
  author={Liu, Xiao and Yin, Da and Zheng, Jingnan and Zhang, Xingjian and Zhang, Peng and Yang, Hongxia and Dong, Yuxiao and Tang, Jie},
  booktitle={KDD'22},
  pages={3418--3428},
  year={2022}
}
```
