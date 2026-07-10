import io

import structlog


def pytest_configure() -> None:
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=io.StringIO()),
        cache_logger_on_first_use=True,
    )
