"""Sample test file to demonstrate pytest markers."""

import pytest


@pytest.mark.unit
def test_sample_unit() -> None:
    """Sample unit test (fast, no external dependencies)."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_sample_integration() -> None:
    """Sample integration test (medium speed, mocked services)."""
    assert True


@pytest.mark.e2e
def test_sample_e2e() -> None:
    """Sample E2E test (slow, real external services).

    This test will be EXCLUDED when running `make test-fast`.
    """
    assert True
