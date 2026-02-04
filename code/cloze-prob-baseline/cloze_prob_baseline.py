# Cloze prob baseline model. 
# This code runs the 4 different alternative strucutres using the 
# pseudo cloze probability measurements from the stimulus generation 
# experiments. 

# ----------------------------------
# Paths 
# ----------------------------------

EXP_DATA = pd.read_csv('../../data/sca_dataframe.csv')
PSEUDO_CLOZE_DATA = pd.read_csv('../../data/inside_the_set/word_freq_and_cloze_prob.csv')

