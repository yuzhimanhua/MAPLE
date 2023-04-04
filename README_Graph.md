# The Graph Format of MAPLE
We create a graph format of MAPLE for graph mining tasks (e.g., node classification, link prediction). You can download the datasets from [**HERE**](https://zenodo.org/record/7797563). Once you unzip the downloaded file, there are 20 folders, corresponding to 20 datasets. Each of the 20 datasets is a heterogeneous graph with 4 types of nodes (i.e., papers, labels, venues, and authors) and 4 types of edges (i.e., paper-paper edges, paper-label edges, paper-venue edges, and paper-author edges). Each node is associated with text information, which can be viewed as node attributes.

## Links

- [Dataset Statistics](#dataset-statistics)
- [Data Format](#data-format)
- [MeSH Nodes in Biology, Chemistry, and Medicine Datasets](#mesh-nodes-in-biology-chemistry-and-medicine-datasets)

## Dataset Statistics
### Node Statistics
| Folder                       | Field                         | #Paper Nodes | #Label Nodes | #Venue Nodes | #Author Nodes |
| ---------------------------- | ----------------------------- | --------- | -------- | ------- | ---------- |
| ```Art```                    | Art                           | 58,373    | 1,990    | 98      | 54,802     |
| ```Philosophy```             | Philosophy                    | 59,296    | 3,758    | 98      | 36,619     |
| ```Geography```              | Geography                     | 73,883    | 3,285    | 98      | 157,423    |
| ```Business```               | Business                      | 84,858    | 2,392    | 97      | 100,525    |
| ```Sociology```              | Sociology                     | 90,208    | 1,935    | 98      | 85,793     |
| ```History```                | History                       | 113,147   | 2,689    | 99      | 84,529     |
| ```Political_Science```      | Political Science             | 115,291   | 4,990    | 98      | 93,393     |
| ```Environmental_Science```  | Environmental Science         | 123,945   | 694      | 100     | 265,728    |
| ```Economics```              | Economics                     | 178,670   | 5,205    | 97      | 135,247    |
| ```CSRankings```             | Computer Science (Conference) | 263,393   | 13,613   | 75      | 331,582    |
| ```Engineering```            | Engineering                   | 270,006   | 10,683   | 100     | 430,046    |
| ```Psychology```             | Psychology                    | 372,954   | 7,641    | 100     | 460,123    |
| ```Computer_Science```       | Computer Science (Journal)    | 410,603   | 15,540   | 96      | 634,506    |
| ```Geology```                | Geology                       | 431,834   | 7,883    | 100     | 471,216    |
| ```Mathematics```            | Mathematics                   | 490,551   | 14,271   | 98      | 404,066    |
| ```Materials_Science```      | Materials Science             | 1,337,731 | 6,802    | 99      | 1,904,549  |
| ```Physics```                | Physics                       | 1,369,983 | 16,664   | 91      | 1,392,070  |
| ```Biology```                | Biology                       | 1,588,778 | 64,267   | 100     | 2,730,547  |
| ```Chemistry```              | Chemistry                     | 1,849,956 | 35,538   | 100     | 2,721,253  |
| ```Medicine```               | Medicine                      | 2,646,105 | 36,619   | 100     | 4,345,385  |

### Edge Statistics
| Folder                       | Field                         | #Paper-Paper Edges | #Paper-Label Edges | #Paper-Venue Edges | #Paper-Author Edges |
| ---------------------------- | ----------------------------- | ---------- | ---------- | --------- | ---------- |
| ```Art```                    | Art                           | 7,184      | 141,542    | 58,373    | 76,728     |
| ```Philosophy```             | Philosophy                    | 63,610     | 220,311    | 59,296    | 66,232     |
| ```Geography```              | Geography                     | 263,331    | 164,799    | 73,883    | 267,861    |
| ```Business```               | Business                      | 797,408    | 301,773    | 84,858    | 199,068    |
| ```Sociology```              | Sociology                     | 267,524    | 209,142    | 90,208    | 143,573    |
| ```History```                | History                       | 30,055     | 219,295    | 113,147   | 137,854    |
| ```Political_Science```      | Political Science             | 165,629    | 358,152    | 115,291   | 172,497    |
| ```Environmental_Science```  | Environmental Science         | 761,972    | 265,040    | 123,945   | 533,108    |
| ```Economics```              | Economics                     | 1,532,072  | 886,380    | 178,670   | 351,288    |
| ```CSRankings```             | Computer Science (Conference) | 1,464,679  | 1,603,672  | 263,393   | 892,758    |
| ```Engineering```            | Engineering                   | 1,212,376  | 1,346,644  | 270,006   | 824,892    |
| ```Psychology```             | Psychology                    | 5,023,328  | 1,876,655  | 372,954   | 1,419,064  |
| ```Computer_Science```       | Computer Science (Journal)    | 1,494,272  | 2,453,776  | 410,603   | 1,353,330  |
| ```Geology```                | Geology                       | 7,628,445  | 2,572,523  | 431,834   | 1,533,938  |
| ```Mathematics```            | Mathematics                   | 2,385,322  | 3,155,560  | 490,551   | 1,055,335  |
| ```Materials_Science```      | Materials Science             | 14,602,443 | 5,923,115  | 1,337,731 | 6,460,288  |
| ```Physics```                | Physics                       | 18,817,355 | 9,960,642  | 1,369,983 | 13,482,670 |
| ```Biology```                | Biology                       | 29,148,006 | 13,196,897 | 1,588,778 | 8,914,115  |
| ```Chemistry```              | Chemistry                     | 20,499,908 | 11,632,581 | 1,849,956 | 7,941,363  |
| ```Medicine```               | Medicine                      | 12,661,657 | 14,224,845 | 2,646,105 | 14,942,938 |


## Data Format
In each folder (e.g., ```Art/```), you can see 8 files. 4 of them contain node information (i.e., ```papers.txt```, ```labels.txt```, ```venues.txt```, and ```authors.txt```), and the other 4 contain edge information (i.e., ```paper-paper.txt```, ```paper-label.txt```, ```paper-venue.txt```, and ```paper-author.txt```).

For node files, each line represent one node. The first column is the node ID; the second column is node text information (i.e., paper title+abstract, label name, venue name, and author name). For example,

**```papers.txt```**:
```
PAPER_2795952975	reading sugar mill ruins the island nobody spoiled and other fantasies of colonial desire
PAPER_2009758184	issues in the levantine epipaleolithic ...
PAPER_2333162778	the life and unusual ideas of adelbert ames jr ...
```

**```labels.txt```**:
```
LABEL_2780583484	papyrus
LABEL_2778949450	scientific writing
LABEL_2780412351	purgatory
```

**```venues.txt```**:
```
VENUE_26308392	the journal of aesthetics and art criticism
VENUE_93676754	modern language review
VENUE_998751717	classical world
```

**```authors.txt```**:
```
AUTHOR_12035	stephen rickerby
AUTHOR_127649	clementine deliss
AUTHOR_1395514	tomas garciasalgado
```

For edge files, each line represent one edge. The first column is the source node ID; the second column is the target node ID. Note that the edge in ```paper-paper.txt``` is directed, where the first node cites the second node. For example,

**```paper-paper.txt```**:
```
PAPER_2009758184	PAPER_2093664100
PAPER_2018118016	PAPER_2324859060
PAPER_2085948119	PAPER_2077568497
```

**```paper-label.txt```**:
```
PAPER_2795952975	LABEL_52119013
PAPER_2009758184	LABEL_15708023
PAPER_2333162778	LABEL_554144382
```

**```paper-venue.txt```**:
```
PAPER_2795952975	VENUE_48411547
PAPER_2009758184	VENUE_173708173
PAPER_2333162778	VENUE_103229351
```

**```paper-author.txt```**:
```
PAPER_2795952975	AUTHOR_2100569728
PAPER_2795952975	AUTHOR_2118309675
PAPER_2009758184	AUTHOR_2591088348
```

## MeSH Nodes in Biology, Chemistry, and Medicine Datasets
In the original format of MAPLE, we have three additional datasets ```Biology_MeSH```, ```Chemistry_MeSH```, and ```Medicine_MeSH```, where each paper has its MeSH labels (different from the labels from the Microsoft Academic Graph mentioned above). When creating the graph format of MAPLE, we merge ```X_MeSH``` and ```X``` datasets together (X = Biology, Chemistry, and Medicine). Therefore, in these three folders, you can see two additional files ```meshs.txt``` and ```paper-mesh.txt``` representing MeSH nodes and Paper-MeSH edges, respectively.

### Node and Edge Statistics
| Folder                       | Field                         | #MeSH Nodes | #Paper-MeSH Edges |
| ---------------------------- | ----------------------------- | ----------- | ----------------- |
| ```Biology```                | Biology                       | 25,039      | 19,131,720        |
| ```Chemistry```              | Chemistry                     | 21,585      | 8,370,930         |       
| ```Medicine```               | Medicine                      | 25,188      | 18,162,351        |

### Data Format

**```meshs.txt```**:
```
MESH_D000818	animals
MESH_D001824	body constitution
MESH_D005075	biological evolution
```

**```paper-mesh.txt```**:
```
PAPER_1816482797	MESH_D005810
PAPER_1816482797	MESH_D005808
PAPER_1816482797	MESH_D020125
```
