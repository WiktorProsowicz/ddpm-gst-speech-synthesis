# ------------------------------------------------------------------------
# This file contains the definition of a docker image used in course
# of the development process. The image built from this config contains
# the basic dependencies one needs in order to be able userto setup the whole
# environment without installing anything locally on the system.
# ------------------------------------------------------------------------

    FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
    ARG DEBIAN_FRONTEND=noninteractive
    RUN useradd -ms /bin/bash devcontainer
   
    # Installing the main packages required for a new project's user
    # They're necessary to correctly use the project's content
    RUN apt update && \
        # Enabling adding apt repository and adding ppas for necessary packages
        apt install -y software-properties-common && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        # Main dependencies
        apt update && apt install -y \
            git nano
    
    USER devcontainer
