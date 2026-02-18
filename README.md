# olmocr

Worker to transcribe some handwritten text

## Docker container

### Create container

To run the worker on your machine, you need first to create a docker container with this command:

``` bash
docker build --tag "olmocr_worker" .
```

### Run container

Don't forget to create a .env file as below:


``` bash
ARKINDEX_API_URL="http://..." # The URL of your arkindex instance
ARKINDEX_API_TOKEN="XXX"      # Token that you have generated in your arkindex profile (in the app)
ARKINDEX_WORKER_RUN_ID="XXX"  # The worker run ID to use
```

Then, please use this command with the correspondent element ID:

``` bash
docker run --rm \
  --env-file .env \
  olmocr_worker \
  worker-olmocr \
  --element="936251ad-d5c4-4486-856f-d6cdc4a47817"
```

### Stop the container

Just use this command:

``` bash
docker stop olmocr_worker
```

## Development

For development and tests purpose it may be useful to install the worker as a editable package with pip.

```shell
pip install -e .
```

## Linter

Code syntax is analyzed before submitting the code.\
To run the linter tools suite you may use pre-commit.

```shell
pip install pre-commit
pre-commit run -a
```

## Run tests

Tests are executed with tox using [pytest](https://pytest.org).

```shell
pip install tox
tox
```

To recreate tox virtual environment (e.g. a dependencies update), you may run `tox -r`

## Deployment

### Worker

1. [Create a worker](https://doc.teklia.com/arkindex/workers/create/#create-a-worker) using the Arkindex frontend. You can set whatever name/slug/type. We'll use:
   - Name: `olmocr`
   - Slug: `olmocr`
   - Type: `transcription`
2. [Import a new version](https://doc.teklia.com/arkindex/workers/create/#create-a-worker-version) for that worker with the following settings:
   - Docker image reference (from [this registry](https://gitlab.teklia.com/workers/olmocr/container_registry)):  `registry.gitlab.teklia.com/workers/olmocr:0.1.0`
   - YAML configuration: Paste the contents of the [arkindex/olmocr.yml](arkindex/olmocr.yml) file found in this repository.
