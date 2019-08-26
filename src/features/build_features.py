from numpy import c_, searchsorted
from sklearn.cluster import KMeans
from featexp import univariate_plotter
from pandas import DataFrame, get_dummies
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
import warnings
warnings.filterwarnings("ignore")
