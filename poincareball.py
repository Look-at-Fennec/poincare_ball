from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath

# Read the sample relations file and train the model
relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
model = PoincareModel(train_data=relations, size=2)
model.train(epochs=50)

import plotly
import gensim.viz.poincare

plotly.offline.init_notebook_mode(connected=False)
prefecutre_map = gensim.viz.poincare.poincare_2d_visualization(model=model,
                                                               tree=relations,
                                                               figure_title="tutorial",
                                                               show_node_labels=model.kv.vocab.keys())
plotly.offline.iplot(prefecutre_map)