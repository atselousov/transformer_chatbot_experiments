FROM jpetazzo/dind

# Install req
RUN apt-get update &&  \
    apt-get -y install openssh-server wget curl rsync vim \
    htop less libsm6 libxext6 libxrender-dev git && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*
    # rm /var/lib/apt/lists/*_* -- to keep container small
    # libsm6 libxext6 and libxrender-dev -- need for correct opencv

RUN mkdir -p /var/run/sshd
RUN mkdir -p /root/.ssh
ADD id_rsa.pub /root/.ssh/id_rsa.pub
RUN cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh/
RUN chmod 600 /root/.ssh/authorized_keys

RUN echo "export PATH=$PATH" >> /etc/profile && \
  echo "ldconfig" >> /etc/profile

EXPOSE 9200
EXPOSE 22

COPY entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

ENTRYPOINT ["bash", "/tmp/entrypoint.sh"]
