FROM debian:bullseye-slim

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-client \
        software-properties-common \
        libgl1-mesa-glx \
        libx11-dev \
        locales \
        tk \
        python3-pip \
        python-is-python3 \
        python3-dev \
        python3-tk \
        make \
        git \
        texlive-base \
        latexmk \
        texlive-latex-extra \
        tex-gyre \
        librsvg2-bin \ 
    && rm -rf /var/lib/apt/lists/*


# to fix locale errors
RUN sed -i 's/# \(en_US\.UTF-8 .*\)/\1/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install some python dependencies
RUN pip install tox genbadge[tests,coverage] numpy matplotlib serums tornado netCDF4 gncpy pytest "setuptools_scm[toml]>=6.2" opencv-python

RUN python3 -m pip install --upgrade pip

RUN pip install --upgrade setuptools

USER $USERNAME