from logging import Logger, getLogger

from arkindex_worker.models import Element
from arkindex_worker.worker import ElementsWorker

logger: Logger = getLogger(__name__)


class Demo(ElementsWorker):
    def process_element(self, element: Element) -> None:
        logger.info(f"Demo processing element ({element.id})")


def main() -> None:
    Demo(description="Worker to transcribe some handwritten text").run()


if __name__ == "__main__":
    main()
