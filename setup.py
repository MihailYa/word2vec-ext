from distutils.core import setup
setup(name='word2vecext',
      description='Word2Vec extensions',
      author='Mihail Yazenok',
      author_email='mihailyazenok@gmail.com',
      url='https://github.com/MihailYa/word2vec-ext',
      version='0.0.2',
      py_modules=['word2vec_ext.word2vec', 'word2vec_ext.ecg_processing', 'word2vec_ext.kmeans'],
      install_requires=[
          "wfdb=3.4.1",
          "sklearn",
          "gensim=4.1.2",
          "pandas=1.1.5",
          "scikit-learn=1.0.1",
          "numpy=1.21.5",
          "matplotlib=3.5.1",
          "py-ecg-detectors=1.1.0"
      ]
      )
