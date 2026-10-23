# Extends the existing project sandbox (built from ../../Dockerfile as
# `bugfix-sandbox`) without changing it: adds a non-root user and the harness.
FROM bugfix-sandbox

RUN useradd --create-home --uid 10001 sandbox
COPY harness.py seccomp_filter.py seccomp-default.json /harness/
RUN chmod 0444 /harness/*

USER sandbox
WORKDIR /tmp
