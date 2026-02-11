import os

import pytest

from arkindex.mock import MockApiClient
from arkindex_worker.worker.base import BaseWorker


@pytest.fixture(autouse=True)
def _setup_environment(responses, monkeypatch) -> None:
    """Setup needed environment variables"""

    # Allow accessing remote API schemas
    # defaulting to the prod environment
    schema_url = os.environ.get(
        "ARKINDEX_API_SCHEMA_URL",
        "https://demo.arkindex.org/api/v1/openapi/?format=openapi-json",
    )
    responses.add_passthru(schema_url)

    # Set schema url in environment
    os.environ["ARKINDEX_API_SCHEMA_URL"] = schema_url
    # Setup a fake worker run ID
    os.environ["ARKINDEX_WORKER_RUN_ID"] = "1234-olmocr"
    # Setup a fake corpus ID
    os.environ["ARKINDEX_CORPUS_ID"] = "1234-corpus-id"

    # Setup a mock api client instead of using a real one
    def mock_setup_api_client(self):
        self.api_client = MockApiClient(schema_url=schema_url)

    monkeypatch.setattr(BaseWorker, "setup_api_client", mock_setup_api_client)
