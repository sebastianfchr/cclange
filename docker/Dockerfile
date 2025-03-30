from nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get install -y python3.9
# re-set symlink to python3.9, not to the included 3.8 
RUN rm /bin/python3 && ln -s /bin/python3.9 /bin/python3
RUN apt-get install -y pip
RUN apt-get install -y uvicorn

RUN python3.9 -m pip install pandas
RUN python3.9 -m pip install tensorflow==2.7.0
# this tf needs lower numpy than what comes with python3.9
RUN python3.9 -m pip install 'numpy<2' 
RUN python3.9 -m pip install -U scikit-learn
RUN python3.9 -m pip install -U transformers==4.27.4
RUN python3.9 -m pip install protobuf==3.20.*
RUN python3.9 -m pip install fastapi
RUN python3.9 -m pip install uvicorn
RUN python3.9 -m pip install websockets==15.0.1

