"""Minimal local compatibility layer for pickled scikit-learn artifacts.

This project stores trained models as pickles. The runtime environment for this
workspace does not have a compatible scikit-learn installation available, so we
provide just enough of the public module structure for joblib to unpickle the
saved model and scaler artifacts.
"""
