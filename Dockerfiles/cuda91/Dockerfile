# FROM nvidia/cuda:9.1-devel
FROM nvidia/cuda:9.1-devel-ubuntu16.04
# RUN ls /usr/local/cuda/include
# FROM nvidia/cuda:9.1-runtime  

# RUN apt-get update \
#   && apt-get install -y python3-pip python3-dev \
#   && cd /usr/local/bin \
#   && ln -s /usr/bin/python3 python \
#   && pip3 install --upgrade pip \
#   && rm -rf /var/lib/apt/lists/*

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6 \
    && conda install pip \
    && conda update conda \
    # && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes
ENV PATH /opt/conda/bin:$PATH
# RUN ls /usr/local/cuda/include
# COPY sample.py requirements.txt ./ 
# RUN pip install -r requirements.txt
# RUN ls .
# CMD python3 sample.py