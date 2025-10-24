FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -q && \
    apt-get install -qy --no-install-recommends \
        build-essential wget perl libfontconfig1 \
        fonts-dejavu fonts-freefont-ttf ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ---- Install TeX Live (basic scheme) ----
RUN wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz && \
    mkdir /install-tl-unx && \
    tar -xzf install-tl-unx.tar.gz -C /install-tl-unx --strip-components=1 && \
    echo "selected_scheme scheme-basic" > /install-tl-unx/texlive.profile && \
    /install-tl-unx/install-tl -profile /install-tl-unx/texlive.profile && \
    rm -r /install-tl-unx install-tl-unx.tar.gz

# ---- Detect install dir and add to PATH ----
RUN export TL_ROOT=$(find /usr/local/texlive -maxdepth 1 -type d -name "20*" | head -n 1) && \
    export TL_BIN="$TL_ROOT/bin/$(ls $TL_ROOT/bin)" && \
    PATH=$TL_BIN:$PATH tlmgr update --self && \
    PATH=$TL_BIN:$PATH tlmgr install \
        collection-xetex \
        latexmk \
        babel babel-russian \
        cm-super \
        hyphen-russian \
        polyglossia fontspec xunicode xecyr \
        extsizes \
        csquotes \
        biblatex biber logreq \
        xcolor \
        wrapfig \
        caption \
        floatrow \
        mathtools \
        esint && \
    ln -s $TL_BIN/* /usr/local/bin/

# ---- Runtime defaults ----
ENV HOME=/data
WORKDIR /data
VOLUME ["/data"]
