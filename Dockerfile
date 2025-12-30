FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Build C++ solver
WORKDIR /app/src/build
RUN cmake .. && make test

# Setup Streamlit app
WORKDIR /app/streamlit
COPY streamlit/ /app/streamlit/

# Copy compiled binary to where the app expects it
RUN mkdir -p bin && cp /app/src/build/test bin/test

# Install Python dependencies using uv
RUN uv pip install --system .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
