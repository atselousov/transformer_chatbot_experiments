FROM ufoym/deepo:pytorch-py36

RUN apt-get update &&  \
    apt-get -y install openssh-server wget curl rsync vim unzip default-jre \
    htop less git && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*
    # rm /var/lib/apt/lists/*_* -- to keep container small

RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /workspace

COPY ./ ./

RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon'); nltk.download('perluniprops')"
RUN git clone https://github.com/facebookresearch/ParlAI.git && cd ParlAI && python setup.py develop
RUN python -m spacy download en
RUN git clone https://github.com/deepmipt/convai_router_bot.git

EXPOSE 9200

# ssh server for pycharm
RUN mkdir -p /var/run/sshd
RUN mkdir -p /root/.ssh
ADD id_rsa.pub /root/.ssh/id_rsa.pub
RUN cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh/
RUN chmod 600 /root/.ssh/authorized_keys

RUN echo "export PATH=$PATH" >> /etc/profile && \
  echo "ldconfig" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
