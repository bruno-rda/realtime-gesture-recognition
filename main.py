import socket
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from realtime import RealTimeTrainer, RealTimePredictor, InterfaceHandler
from emg_processing import ManualProcessor
from feature_extraction import ManualFeatureExtractor
from config import get_settings
import threading
import queue
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

pipeline = Pipeline([
    ('feature_selector', SelectPercentile(f_classif, percentile=settings.feature_selector_percentile)),
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
    base_dir=settings.base_dir,
)

predictor = RealTimePredictor(
    pipeline=pipeline,
    processor=processor,
    window_size=settings.window_size,
    step_size=settings.step_size,
    sampling_rate=settings.sampling_rate,
)

interface = InterfaceHandler(trainer, predictor)

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


if __name__ == '__main__':
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
    interface.start()

    try:
        while True:
            pkt = data_queue.get()

            if len(pkt) % 8 != 0:
                logger.warning(f'{len(pkt)} bytes (excess {len(pkt) % 8}); truncating.')
                pkt = pkt[:-len(pkt) % 8]

            data = np.frombuffer(pkt, dtype='<f8')
            n_package_samples = len(data) // settings.n_channels
            data = np.array(data).reshape(n_package_samples, settings.n_channels)

            # Update the controller with the new data
            interface.update(data)

    except KeyboardInterrupt:
        logger.info('Keyboard interrupt detected. Exiting...')
        
    except Exception as e:
        logger.error(f'Error: {e}')
    
    finally:
        logger.info('Shutting down...')
        stop_event.set()
        sock.close()
        receiver_thread.join()
        interface.stop()