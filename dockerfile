FROM python:3.6.0
WORKDIR /app
COPY setup.py setup.py
RUN pip3 install .

COPY . .
CMD ["python3", "-m", "example.example2_predicting_surrounding_words"]