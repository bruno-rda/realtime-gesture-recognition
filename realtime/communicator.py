import serial
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class SerialCommunicator:
    def __init__(
        self, 
        port: str, 
        baudrate: int = 9600, 
        timeout: float = 1,
        chunk_size: int = 1,
        message_mapping: Optional[dict[Any, Any]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.message_mapping = message_mapping or dict()

        self.serial_connection: Optional[serial.Serial] = None
        self.chunk_buffer: list[bytes] = []
        self._connection_warned = False

    @property
    def is_active(self):
        return (
            self.serial_connection is not None and 
            self.serial_connection.is_open
        )

    def open(self):
        self.serial_connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout
        )

        logger.info(f'Opened serial connection to {self.port} at {self.baudrate} baud.')

    def close(self):
        if self.is_active:
            self.serial_connection.close()
            self.serial_connection = None
            logger.info('Serial connection closed.')

    def send(self, message: str) -> None:
        if not self.is_active:
            self._warn_once()
            return

        self._restore_warning()

        # Apply message mapping
        mapped = self.message_mapping.get(message, message)
        self.chunk_buffer.append(mapped.encode('utf-8'))

        if len(self.chunk_buffer) >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self.is_active or not self.chunk_buffer:
            return

        self.serial_connection.write(b'\n'.join(self.chunk_buffer) + b'\n')
        self.chunk_buffer.clear()

    def _warn_once(self) -> None:
        if not self._connection_warned:
            logger.warning('Serial connection not open.')
            self._connection_warned = True

    def _restore_warning(self) -> None:
        if self._connection_warned:
            logger.info('Serial connection restored.')
            self._connection_warned = False