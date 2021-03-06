FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install -y vim \
  && apt-get install -y unzip \
  && apt-get install -y default-jre \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get clean && apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /workspace

COPY ./ ./

RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon'); nltk.download('perluniprops'); nltk.download('punkt')"
RUN git clone https://github.com/facebookresearch/ParlAI.git && cd ParlAI && python setup.py develop
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN python -m spacy download en
RUN cd metrics/3rdparty && wget http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz && tar -xvzf ./meteor-1.5.tar.gz && rm meteor-1.5.tar.gz

EXPOSE 9200

ENTRYPOINT ["/bin/bash"]
