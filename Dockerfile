FROM python:3.9

RUN git clone https://github.com/jeanphilippemercier/nlloc.git
RUN apt update; apt install -y gcc-9; cp /usr/bin/gcc-9 /usr/bin/gcc
RUN cd nlloc; make
RUN cp nlloc/Grid2GMT /usr/bin/.; cp nlloc/Grid2Time /usr/bin/.; cp nlloc/Loc2ddct /usr/bin/.
RUN cp nlloc/LocSum /usr/bin/.
RUN cp nlloc/NLLoc /usr/bin/.; cp nlloc/PhsAssoc /usr/bin/.; cp nlloc/Time2Angles /usr/bin/.
RUN cp nlloc/Time2EQ /usr/bin/.; cp nlloc/Vel2Grid /usr/bin/.
RUN cp nlloc/fmm2grid /usr/bin/.; cp nlloc/fpfit2hyp /usr/bin/.; cp nlloc/oct2grid /usr/bin/.
RUN cp nlloc/scat2latlon /usr/bin/.

COPY . useis
RUN cd useis; pip install poetry; poetry install

WORKDIR useis
ENTRYPOINT poetry shell
