FROM python:3.12

WORKDIR /code

COPY ./pyproject.toml /code/pyproject.toml

RUN pip install --no-cache-dir --upgrade .

COPY . .
ENV NUMBA_CACHE_DIR=/tmp

EXPOSE 8080

CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
