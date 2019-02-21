FROM ufoym/deepo:pytorch-py36-cu90

RUN apt-get update &&  \
    apt-get -y install openssh-server wget curl rsync vim unzip default-jre tmux \
    htop less git && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*
    # rm /var/lib/apt/lists/*_* -- to keep container small

RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN echo 1122

WORKDIR /workspace

COPY ./ ./
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon'); nltk.download('perluniprops'); nltk.download('punkt')"
RUN git clone https://github.com/facebookresearch/ParlAI.git && cd ParlAI && python setup.py develop
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN python -m spacy download en
#RUN git clone https://github.com/deepmipt/convai_router_bot.git

# Install openssh-server and few useful things
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install openssh-server wget curl rsync vim psmisc procps git && apt-get clean && rm /var/lib/apt/lists/*_*
RUN mkdir -p /var/run/sshd

# Prepare single root user
RUN mkdir -p /root/.ssh
RUN mkdir -p /var/run/sshd

RUN curl -o /root/.ssh/id_rsa.pub -O https://github.com/truskovskiyk.keys
RUN cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys
RUN curl https://github.com/rauf-kurbanov.keys >> /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh/
RUN chmod 600 /root/.ssh/authorized_keys

# configure profile to see all CUDA libraries
RUN echo "export PATH=$PATH" >> /etc/profile &&  echo "ldconfig" >> /etc/profile
# add run script
CMD ["/usr/sbin/sshd", "-D"]
# expose port
EXPOSE 22
EXPOSE 9200




