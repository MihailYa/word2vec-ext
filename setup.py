from distutils.core import setup
setup(name='word2vecext',
      description='Word2Vec extensions',
      author='Mihail Yazenok',
      author_email='mihailyazenok@gmail.com',
      url='https://github.com/MihailYa/word2vec-ext',
      version='0.0.1',
      py_modules=['word2vec_ext.word2vec', 'word2vec_ext.ecg_processing', 'word2vec_ext.kmeans'],
      install_requires=[
          "wfdb",
          "sklearn",
          "gensim==3.8.3",
          "pandas",
          "scikit-learn",
          "numpy",
          "matplotlib",
          "py-ecg-detectors"
      ]
      )
