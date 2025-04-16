#__copyright__   = "Copyright 2024, VISA Lab"
#__license__     = "MIT"

# # Define global args
# ARG FUNCTION_DIR="/home/app/"
# ARG RUNTIME_VERSION="3.8"
# ARG DISTRO_VERSION="3.12"

# FROM alpine:latest
# FROM python:${RUNTIME_VERSION} AS python-alpine

# # Upgrade pip
# RUN python${RUNTIME_VERSION} -m pip install --upgrade pip

# FROM python-alpine AS build-image

# # Include global args in this stage of the build
# ARG FUNCTION_DIR
# ARG RUNTIME_VERSION

# # Create function directory
# RUN mkdir -p ${FUNCTION_DIR}

# # Install Lambda Runtime Interface Client for Python
# RUN python${RUNTIME_VERSION} -m pip install awslambdaric --target ${FUNCTION_DIR}

# # Stage 3 - final runtime image
# # Grab a fresh copy of the Python image
# FROM python-alpine

# # Include global arg in this stage of the build
# ARG FUNCTION_DIR

# # Set working directory to function root directory
# WORKDIR ${FUNCTION_DIR}

# # Copy in the built dependencies
# COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# # Add Lambda Runtime Interface Emulator for local testing
# ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
# RUN chmod 755 /usr/bin/aws-lambda-rie

# # Install ffmpeg using Alpine's package manager
# RUN apt-get update && apt-get install -y ffmpeg

# # Copy and install dependencies from requirements.txt
# COPY requirements.txt ${FUNCTION_DIR}
# RUN python${RUNTIME_VERSION} -m pip install -r requirements.txt --target ${FUNCTION_DIR}

# # Copy the entry script and handler
# COPY entry.sh /entry.sh
# COPY handler.py ${FUNCTION_DIR}

# # Ensure entry.sh has execute permissions
# RUN chmod +x /entry.sh

# # Set ENTRYPOINT and CMD
# ENTRYPOINT [ "/entry.sh" ]
# CMD [ "handler.handler" ]

FROM public.ecr.aws/lambda/python:3.8

# Install tar, xz, and download and install ffmpeg
RUN yum install -y tar xz && \
    curl -L -o ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    mkdir -p /ffmpeg_temp && \
    tar -xJf ffmpeg-release-amd64-static.tar.xz -C /ffmpeg_temp && \
    mv $(find /ffmpeg_temp -name 'ffmpeg') /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg && \
    rm -rf ffmpeg-release-amd64-static.tar.xz /ffmpeg_temp

# Copy function code and requirements
COPY handler.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip3 install -r ${LAMBDA_TASK_ROOT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler
CMD [ "handler.handler" ]




