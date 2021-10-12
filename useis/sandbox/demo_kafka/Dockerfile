FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update && apt-get -y install git && apt-get -y install make && \
    apt-get -y install gcc g++ && \
    git clone https://github.com/jeanphilippemercier/nlloc.git && \
    cd nlloc && make && cp NLLoc /bin/NLLoc && chmod 755 /bin/NLLoc && \
    apt-get -y remove gcc g++ make && cd ..

RUN git clone https://github.com/jeanphilippemercier/uquake-useis.git && \
    cd uquake-useis && apt-get -y install python3 && apt-get -y install pip && \
    apt-get -y install python3.8-venv && python3 -m venv /venv/useis

RUN pip install poetry

RUN  bash /venv/useis/bin/activate && cd /uquake-useis && poetry config virtualenvs.create false &&poetry install
