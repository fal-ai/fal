FROM python:3.13-slim-trixie AS crispy-builder

ARG CRISPY_DOOM_VERSION=7.1
ARG CRISPY_DOOM_SHA256=f0eb02afb81780165ddc81583ed5648cbee8b3205bcc27e181b3f61eb26f8416

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        gcc \
        libsamplerate0-dev \
        libsdl2-dev \
        libsdl2-mixer-dev \
        libsdl2-net-dev \
        make \
        pkg-config \
    && curl -fsSL \
        "https://github.com/fabiangreffrath/crispy-doom/archive/refs/tags/crispy-doom-${CRISPY_DOOM_VERSION}.tar.gz" \
        -o /tmp/crispy-doom.tar.gz \
    && echo "${CRISPY_DOOM_SHA256}  /tmp/crispy-doom.tar.gz" \
        | sha256sum --check --strict \
    && mkdir /tmp/crispy-doom \
    && tar -xzf /tmp/crispy-doom.tar.gz \
        -C /tmp/crispy-doom --strip-components=1 \
    && cmake -S /tmp/crispy-doom -B /tmp/crispy-doom/build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/crispy-doom/build --parallel \
    && install -Dm755 \
        /tmp/crispy-doom/build/src/crispy-doom \
        /opt/crispy-doom/bin/crispy-doom \
    && install -Dm644 \
        /tmp/crispy-doom/COPYING.md \
        /opt/crispy-doom/share/licenses/COPYING.md \
    && install -Dm644 \
        /tmp/crispy-doom.tar.gz \
        /opt/crispy-doom/share/source/crispy-doom-7.1.tar.gz \
    && test -x /opt/crispy-doom/bin/crispy-doom

FROM python:3.13-slim-trixie

RUN sed -i \
        's/Components: main/Components: main contrib non-free non-free-firmware/' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        freedoom=0.13.0-2 \
        gir1.2-gst-plugins-bad-1.0 \
        gstreamer1.0-nice \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-pulseaudio \
        gstreamer1.0-tools \
        gstreamer1.0-x \
        libnvrtc12=12.4.127~12.4.1-2 \
        libsamplerate0 \
        libsdl2-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libsdl2-net-2.0-0 \
        libx11-6 \
        libxtst6 \
        pulseaudio \
        python3-gi \
        python3-gst-1.0 \
        xdotool \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s libnvrtc.so.12 /usr/lib/x86_64-linux-gnu/libnvrtc.so

COPY --from=crispy-builder /opt/crispy-doom /opt/crispy-doom

ENV PATH="/opt/crispy-doom/bin:/usr/games:${PATH}"
ENV PYTHONPATH="/usr/lib/python3/dist-packages"
