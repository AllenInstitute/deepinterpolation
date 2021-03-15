FROM continuumio/miniconda3:4.9.2

ARG REPO_TAG="feature/ephys_transfer"
ARG COMMIT="unspecified to docker build"

RUN adduser deepuser
USER deepuser

ENV REPOS=/home/deepuser/repos
ENV REPO=/home/deepuser/repos/deepinterpolation
ENV CONDA_ENVS_PATH=/home/deepuser/envs
ENV CONDA_PKGS_DIRS=/home/deepuser/pkgs
ENV CONDA_ENV_NAME=deep_env
ENV DEEPINTERPOLATION_COMMIT_SHA=${COMMIT}
ENV TMPDIR=/tmp

RUN mkdir ${REPOS} \
    && git clone -b ${REPO_TAG} https://github.com/AllenInstitute/deepinterpolation ${REPO} \
    && conda create -n ${CONDA_ENV_NAME} python=3.8 \
    && conda run -n ${CONDA_ENV_NAME} pip install ${REPO}

ENTRYPOINT ["/bin/bash", "-c", "$@", "--"]
