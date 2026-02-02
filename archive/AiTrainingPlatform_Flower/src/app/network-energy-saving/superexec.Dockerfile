FROM flwr/superexec:1.25.0

WORKDIR /app

# Create model storage directory
RUN mkdir -p /app/models
RUN mkdir -p /app/results

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]