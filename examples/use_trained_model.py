"""Load pretrained model and transform a sample data."""
from neuroharmony import fetch_trained_model, fetch_sample

neuroharmony = fetch_trained_model()
X = fetch_sample()
x_harmonized = neuroharmony.transform(X)
