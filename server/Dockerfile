FROM ghcr.io/deephaven/server:${VERSION:-0.15.0}
# pip install any of the plugins required on the server
RUN pip install deephaven-plugin-matplotlib
RUN pip install --force-reinstall --no-deps deephaven-plugin-matplotlib
