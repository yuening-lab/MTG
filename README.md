## MTG
This is the source code for paper ''[Text-enhanced Multi-Granularity Temporal Graph
Learning for Event Prediction](https://ieeexplore.ieee.org/document/10027692)'' appeared in the proceedings of ICDM 2022 (regular paper). 

Xiaoxue Han, [Yue Ning](https://yue-ning.github.io/)

### Data
- [ICEWS event data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075 "ICEWS event data") is available online.
- ICEWS news data has not been released publicly. 

### Prerequisites
The code has been successfully tested in the following environment. (For older versions, you may need to modify the code)
- Python 3.8.3
- PyTorch 1.10.0
- dgl 0.8.1
- Sklearn 0.23.1
- numpy 1.18.5

### Sample dataset
- **THA** (Event and news data in Bangkok, Thailand from 2010 to 2016) [OneDrive](https://stevens0-my.sharepoint.com/:f:/g/personal/xhan26_stevens_edu/Eoyh5QyFRjxLivWLB7ejJlMB_zVOuJgBPdBfTXMYQEX7hw)

### Training and testing
Please run the following commands for training and testing. We take the dataset `THA` as the example.
```python
python train.py --data 'THA' -lt 1 -pw 1 -hw 7 --n_runs 1 
```
## Cite

Please cite our paper if you find this code useful for your research:

```
X. Han and Y. Ning, "Text-enhanced Multi-Granularity Temporal Graph Learning for Event Prediction," 2022 IEEE International Conference on Data Mining (ICDM), Orlando, FL, USA, 2022, pp. 171-180, doi: 10.1109/ICDM54844.2022.00027.
```

**BibTeX**
```
@INPROCEEDINGS{10027692,
  author={Han, Xiaoxue and Ning, Yue},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)}, 
  title={Text-enhanced Multi-Granularity Temporal Graph Learning for Event Prediction}, 
  year={2022},
  volume={},
  number={},
  pages={171-180},
  doi={10.1109/ICDM54844.2022.00027}}

```
