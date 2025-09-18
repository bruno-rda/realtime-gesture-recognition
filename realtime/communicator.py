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
        message_mapping: Optional[dict[Any, Any]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.message_mapping = message_mapping or dict()
        self.serial_connection = None
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

    def send(self, message: str):
        if self.is_active:
            if self._connection_warned:
                logger.info('Serial connection restored.')
                self._connection_warned = False

            mapped_message = self.message_mapping.get(message, message)
            self.serial_connection.write(mapped_message.encode('utf-8'))
        else:
            if not self._connection_warned:
                logger.warning('Serial connection not open.')
                self._connection_warned = True