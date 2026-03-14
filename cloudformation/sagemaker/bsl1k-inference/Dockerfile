FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-gpu-py310-cu118-ubuntu20.04-sagemaker

# Copy inference code
COPY inference.py /opt/ml/code/inference.py
COPY model.py /opt/ml/code/model.py
COPY requirements.txt /opt/ml/code/requirements.txt

# Install additional dependencies (most are already in the base image)
RUN pip install --no-cache-dir scipy opencv-python-headless

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENTRYPOINT ["python", "-m", "sagemaker_pytorch_serving_container.serving"]
