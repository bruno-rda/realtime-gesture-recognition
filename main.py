import socket
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from realtime import RealTimeTrainer, RealTimePredictor, SerialCommunicator, InterfaceHandler
from emg_processing import ManualProcessor
from feature_extraction import ManualFeatureExtractor
from config import settings, Settings
import threading
import queue
import logging

logging.basicConfig(
    level=settings.log_level,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def receive_packets(sock, queue, stop_event):
    logger.info('Starting receiver thread')
    while not stop_event.is_set():
        try:
            pkt, _ = sock.recvfrom(65535)
            queue.put(pkt)
        except socket.timeout:
            continue
        except Exception as e:
            logger.error(f'Error receiving packet: {e}')
            break

def process_packet(pkt: bytes, n_channels: int) -> np.ndarray:
    if len(pkt) % 8 != 0:
        logger.warning(f'{len(pkt)} bytes (excess {len(pkt) % 8}); truncating.')
        pkt = pkt[:-len(pkt) % 8]

    data = np.frombuffer(pkt, dtype='<f8')
    n_package_samples = len(data) // n_channels
    data = np.array(data).reshape(n_package_samples, n_channels)
    return data


def create_app(settings: Settings):
    # If trainer_path is provided, the trainer will be loaded from the path.
    # Otherwise, a new trainer will be created.
    if settings.trainer_path is not None:
        trainer = RealTimeTrainer.from_path(settings.trainer_path)
    else:
        pipeline = Pipeline([
            ('feature_selector', SelectPercentile(
                f_classif, 
                percentile=settings.feature_selector_percentile
            )),
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                objective='multi:softmax', 
                seed=settings.model_seed,
                learning_rate=settings.model_learning_rate,
                max_depth=settings.model_max_depth,
                n_estimators=settings.model_n_estimators,
            ))
        ])

        processor = ManualProcessor(ManualFeatureExtractor(simple=True))
        trainer = RealTimeTrainer(
            pipeline=pipeline,
            processor=processor,
            window_size=settings.window_size,
            step_size=settings.step_size,
            sampling_rate=settings.sampling_rate,
            cross_validate=settings.cross_validate,
            should_save=settings.should_save,
            base_dir=settings.experiments_base_dir,
        )

    predictor = RealTimePredictor(
        pipeline=trainer.pipeline,
        processor=trainer.processor,
        window_size=trainer.window_size,
        step_size=trainer.step_size,
        sampling_rate=trainer.sampling_rate,
    )

    communicator = SerialCommunicator(
        port=settings.serial_port,
        baudrate=settings.serial_baudrate,
        timeout=settings.serial_timeout,
        message_mapping=settings.message_mapping,
    )

    interface = InterfaceHandler(
        trainer=trainer, 
        predictor=predictor,
        communicator=communicator,
        show_probs=settings.show_probs,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((settings.udp_ip, settings.udp_port))
    sock.settimeout(1)
    
    data_queue = queue.Queue()
    stop_event = threading.Event()
    receiver_thread = threading.Thread(
        target=receive_packets, 
        args=(sock, data_queue, stop_event), 
        daemon=True
    )
    receiver_thread.start()
    
    return interface, data_queue, stop_event, sock, receiver_thread


if __name__ == '__main__':
    interface, data_queue, stop_event, sock, receiver_thread = create_app(settings)
    
    interface.start()

    try:
        while True:
            pkt = data_queue.get()
            data = process_packet(pkt, settings.n_channels)
            interface.update(data)

    except KeyboardInterrupt:
        logger.info('Keyboard interrupt detected. Exiting...')
        
    except Exception as e:
        logger.error(f'Error: {e}')
    
    logger.info('Shutting down...')
    stop_event.set()
    sock.close()
    receiver_thread.join()
    interface.stop()