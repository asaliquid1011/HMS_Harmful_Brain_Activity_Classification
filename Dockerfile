FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
RUN apt-get update -qq -o=Dpkg::Use-Pty=0 && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-setuptools

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

WORKDIR /workspace
RUN pip3 install torch torchvision torchaudio
ADD requirements.txt /workspace/requirements.txt
RUN pip3 install -r requirements.txt


ADD run.sh /opt/run.sh
RUN chmod 700 /opt/run.sh
CMD /opt/run.sh
