# Extend the official Deephaven Community Core server image with this
# project's extra Python dependencies (yfinance, pandas).
#
# Docs: https://deephaven.io/core/docs/how-to-guides/install-and-run/
FROM ghcr.io/deephaven/server:latest

# Install additional Python packages into the image's Python environment so
# they are importable from the Deephaven query engine / IDE.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
