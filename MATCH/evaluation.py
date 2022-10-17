import warnings
warnings.filterwarnings('ignore')

import click
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from deepxml.evaluation import get_p_1, get_p_3, get_p_5, get_n_3, get_n_5

@click.command()
@click.option('-r', '--results', type=click.Path(exists=True), help='Path of results.')
@click.option('-t', '--targets', type=click.Path(exists=True), help='Path of targets.')
@click.option('--train-labels', type=click.Path(exists=True), default=None, help='Path of labels for training set.')
@click.option('-a', type=click.FLOAT, default=0.55, help='Parameter A for propensity score.')
@click.option('-b', type=click.FLOAT, default=1.5, help='Parameter B for propensity score.')
def main(results, targets, train_labels, a, b):
	res, targets = np.load(results, allow_pickle=True), np.load(targets, allow_pickle=True)
	mlb = MultiLabelBinarizer(sparse_output=True)
	targets = mlb.fit_transform(targets)

	p1, p3, p5, n3, n5 = get_p_1(res, targets, mlb), get_p_3(res, targets, mlb), get_p_5(res, targets, mlb), \
						 get_n_3(res, targets, mlb), get_n_5(res, targets, mlb)
	print('P@1:', p1, ', ', \
		  'P@3:', p3, ', ', \
		  'P@5:', p5, ', ', \
		  'NDCG@3:', n3, ', ', \
		  'NDCG@5:', n5)

	with open('scores.txt', 'a') as fout:
		fout.write('{:.4f}'.format(p1)+'\t'+'{:.4f}'.format(p3)+'\t'+'{:.4f}'.format(p5)+'\t'+ \
				   '{:.4f}'.format(n3)+'\t'+'{:.4f}'.format(n5)+'\n')
	
if __name__ == '__main__':
	main()
