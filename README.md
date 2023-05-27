# The Effect of Metadata on Scientific Literature Tagging: A Cross-Field Cross-Model Study

[![License: Open Data Commons Attribution](https://img.shields.io/badge/License-ODC_BY-brightgreen.svg)](https://opendatacommons.org/licenses/by/)

This repository contains the datasets and source code used in our paper [The Effect of Metadata on Scientific Literature Tagging: A Cross-Field Cross-Model Study](https://arxiv.org/abs/2302.03341).

## Links

- [Datasets](#datasets)
- [Additional Datasets with MeSH Labels](#additional-datasets-with-mesh-labels)
- [Running Parabel](#running-parabel)
- [Running Transformer](#running-transformer)
- [Running OAG-BERT](#running-oag-bert)
- [References](#references)

## Datasets
**NOTE: If you are working on graph mining tasks (e.g., node classification, link prediction) in homogeneous/heterogeneous/attributed/text-rich networks, we have also created a graph format of MAPLE, and you can refer to [README_Graph.md](README_Graph.md) for more details.**

The MAPLE benchmark constructed by us contains 20 datasets across 19 fields for scientific literature tagging. You can download the datasets from [**HERE**](https://doi.org/10.5281/zenodo.7611544). Once you unzip the downloaded file, you can see a folder ```MAPLE/```. Please put the folder under the main directory ```./``` of this code repository.

There are 23 folders under ```MAPLE/```, corresponding to 23 datasets. 20 of them with MAG labels are mentioned in the main text of our paper; the other 3 datasets with MeSH labels will be introduced in [the next section](#additional-datasets-with-mesh-labels). Statistics of the 20 "main" datasets are as follows:

### Dataset Statistics

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

### Data Format

In each folder (e.g., ```Art/```), you can see four files: ```authors.txt```, ```labels.txt```, ```papers.json```, and ```venues.txt```.

```authors.txt``` has 3 columns: author id, normalized author name, and original author name:
```
12035	stephen rickerby	Stephen Rickerby
127649	clementine deliss	Clementine Deliss
1395514	tomas garciasalgado	Tomás García-Salgado
...
```

```venues.txt``` has 3 columns: venue id, normalized venue name, and original venue name:
```
26308392	the journal of aesthetics and art criticism	The Journal of Aesthetics and Art Criticism
93676754	modern language review	Modern Language Review
998751717	classical world	Classical World
...
```

```labels.txt``` has 3 columns: label id, label name, and depth of the label (1-5, with 1 being the coarsest and 5 being the finest):
```
2780583484	papyrus	2
2778949450	scientific writing	2
2780412351	purgatory	2
...
```

```papers.json``` has text and metadata information of each paper. Each line is a json record representing one paper. For example,
```
{
  "paper": "2333162778",
  "venue": "103229351",
  "year": "1987",
  "title": "the life and unusual ideas of adelbert ames jr",
  "label": [
    "554144382", "153349607"
  ],
  "author": [
    "2162173344"
  ],
  "reference": [
    "132232344", "378964350", "562124327", ...
  ],
  "abstract": "this paper is a summary of the life and major achievements of adelbert ames jr an american ...",
  "title_raw": "The Life and Unusual Ideas of Adelbert Ames, Jr.",
  "abstract_raw": "This paper is a summary of the life and major achievements of Adelbert Ames, Jr., an American ..."
}
```

## Additional Datasets with MeSH Labels
The three additional datasets: ```Biology_MeSH```, ```Chemistry_MeSH```, and ```Medicine_MeSH``` are constructed from ```Biology```, ```Chemistry```, and ```Medicine```, respectively, by obtaining the MeSH labels of each paper (and removing those papers without MeSH labels).

### Dataset Statistics

| Folder                       | Field                         | #Papers  | #Labels  | #Venues | #Authors  | #References   |
| ---------------------------- | ----------------------------- | -------- | -------- | ------- | --------- | ------------- |
| ```Biology_MeSH```           | Biology-MeSH                  | 1,379,393 | 25,039   | 100     | 2,486,814 | 6,876,739     |
| ```Chemistry_MeSH```         | Chemistry-MeSH                | 762,129   | 21,585   | 87      | 1,498,358 | 5,928,908     |
| ```Medicine_MeSH```          | Medicine-MeSH                 | 1,536,660 | 25,188   | 100     | 2,791,165 | 7,190,021     |

### Data Format

In each folder (e.g., ```Biology_MeSH/```), you can see five files: ```authors.txt```, ```labels.txt```, **```labels_mesh.txt```**, ```papers.json```, and ```venues.txt```.

```authors.txt``` and ```venues.txt``` have the same format as in the 20 "main" datasets.

```labels.txt``` has 2 columns: MeSH label id and original MeSH label name:
```
D000818	Animals
D001824	Body Constitution
D005075	Biological Evolution
...
```

```labels_mesh.txt``` has >=2 columns: MeSH label id, normalized MeSH label name, and all entry terms (i.e., synonyms) of the MeSH label:
```
D000818	animals	animalia
D001824	body constitution	body constitutions	constitution body	constitutions body
D005075	biological evolution	evolution biological
...
```

```papers.json``` has the same format as in the 20 "main" datasets. The only difference is that the "label" field now contains all MeSH labels of the paper. For example,
```
{
  "paper": "1816482797",
  "venue": "166515463",
  "year": "2015",
  "title": "proteins linked to autosomal dominant and autosomal recessive disorders harbor characteristic ...",
  "label": [
    "D005810", "D005808", "D020125", "D019295", "D030541", ...
  ],
  "author": [
    "2303839782", "2953263946", "2160643821", ...
  ],
  "reference": [
    "80748578", "1563940013", "1570281893", ...
  ],
  "abstract": "the role of rare missense variants in disease causation remains difficult to interpret ...",
  "title_raw": "Proteins linked to autosomal dominant and autosomal recessive disorders harbor characteristic ...",
  "abstract_raw": "The role of rare missense variants in disease causation remains difficult to interpret ..."
}
```

## Running Parabel
The code of Parabel is written in C++. It is adapted from [**the original implementation**](http://manikvarma.org/code/Parabel/download.html) by Prabhu et al. You need to run the following script.
```
cd ./Parabel/
./run.sh
```
P@_k_ and NDCG@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./Parabel/scores.txt```. The prediction results can be found in ```./Parabel/Sandbox/Results/{dataset}/score_mat.txt```.

## Running Transformer
**GPUs are required. We use one NVIDIA GeForce GTX 1080 Ti GPU in our experiments.** 

The code of Transformer is written in Python 3.6. It is adapted from [**the original implementation**](https://github.com/XunGuangxu/CorNet) by Xun et al. You need to first install the dependencies like this:
```
cd ./Transformer/
pip3 install -r requirements.txt
```
Then, you need to download [**the GloVe embeddings**](https://drive.google.com/file/d/1GNmoqocua51496sP8cr86hX-mOyR-ExW/view?usp=share_link) (originally from [here](https://nlp.stanford.edu/projects/glove/)). Once you unzip the downloaded file, please put it (i.e., the ```data/``` folder) under ```./Transformer/```. Then, you can run the code.
```
./run.sh
```
P@_k_ and NDCG@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./Transformer/scores.txt```. The prediction results can be found in ```./Transformer/predictions.txt```.

## Running OAG-BERT
**GPUs are required. We use one NVIDIA GeForce GTX 1080 Ti GPU in our experiments.** 

The code of OAG-BERT is written in Python 3.7. It is adapted from [**the original implementation**](https://github.com/THUDM/OAG-BERT) by Liu et al. You need to first install PyTorch >= 1.7.1, and then the [**CogDL**](https://github.com/THUDM/cogdl) package. These two steps can be done by running the following:
```
cd ./OAGBERT/
./setup.sh
```
Then, you can run the code.
```
./run.sh
```
P@_k_ and NDCG@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./OAGBERT/Parabel/scores.txt```. The prediction results can be found in ```./OAGBERT/Parabel/Sandbox/Results/{dataset}/score_mat.txt```.

## References
If you find the MAPLE benchmark or this repository useful, please cite our paper:
```
@inproceedings{zhang2023effect,
  title={The effect of metadata on scientific literature tagging: A cross-field cross-model study},
  author={Zhang, Yu and Jin, Bowen and Zhu, Qi and Meng, Yu and Han, Jiawei},
  booktitle={WWW'23},
  pages={1626--1637},
  year={2023}
}
```
The MAPLE benchmark is constructed from the Microsoft Academic Graph:
```
@inproceedings{sinha2015overview,
  title={An overview of microsoft academic service (mas) and applications},
  author={Sinha, Arnab and Shen, Zhihong and Song, Yang and Ma, Hao and Eide, Darrin and Hsu, Bo-June and Wang, Kuansan},
  booktitle={WWW'15},
  pages={243--246},
  year={2015}
}
```
The three classifiers in this repository are from the following three papers:
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
