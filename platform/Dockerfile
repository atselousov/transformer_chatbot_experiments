FROM ufoym/deepo:pytorch-py36-cu90

# General library
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

# SSH server
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install openssh-server wget curl rsync vim psmisc procps git && apt-get clean && rm /var/lib/apt/lists/*_*
RUN mkdir -p /var/run/sshd

RUN mkdir -p /root/.ssh
RUN mkdir -p /var/run/sshd

RUN curl https://github.com/truskovskiyk.keys >> /root/.ssh/authorized_keys
RUN curl https://github.com/rauf-kurbanov.keys >> /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh/
RUN chmod 600 /root/.ssh/authorized_keys

# configure profile to see all CUDA libraries
RUN echo "export PATH=$PATH" >> /etc/profile &&  echo "ldconfig" >> /etc/profile

# code requirements
WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon'); nltk.download('perluniprops'); nltk.download('punkt')"
RUN git clone https://github.com/facebookresearch/ParlAI.git && cd ParlAI && python setup.py develop
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN python -m spacy download en
RUN if [ ! -d "convai_router_bot" ]; then git clone https://github.com/deepmipt/convai_router_bot.git; fi

RUN cpan XML::XPath
RUN cpan XML::Twig
RUN cpan Sort::Naturally
RUN cpan String::Util


COPY ./ ./

# add run script
CMD ["/usr/sbin/sshd", "-D"]
# expose port
EXPOSE 22
EXPOSE 8080





