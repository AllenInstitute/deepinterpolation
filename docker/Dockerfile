FROM continuumio/miniconda3:4.8.2

ARG REPO_TAG="feature/ephys_transfer"
ARG COMMIT="unspecified to docker build"

ENV REPOS=/repos
ENV REPO=/repos/deepinterpolation
ENV ENVS=/envs
ENV CONDA_ENV=${ENVS}/deep_env
ENV DEEPINTERPOLATION_COMMIT_SHA=${COMMIT}

RUN mkdir ${REPOS} ${ENVS} \
    && apt-get -y update \
    && git clone -b ${REPO_TAG} https://github.com/AllenInstitute/deepinterpolation ${REPO} \
    && conda create --prefix ${CONDA_ENV} python=3.8 \
    && conda run --prefix ${CONDA_ENV} pip install ${REPO}

ENTRYPOINT ["conda", "run", "--prefix", "${CONDA_ENV}", "$@"]
