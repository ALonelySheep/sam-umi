from typing import List, Optional, Union, Dict, Callable
import numbers
import copy
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.video_recorder import VideoRecorder
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QWidget
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class PointAnnotator(QWidget):
    point_selected = pyqtSignal(float, float)  # Signal to emit when point is selected

    def __init__(self, first_frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Initial Point")
        self.first_frame = first_frame
        
        # Setup UI
        layout = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel()
        self.display_image(first_frame)
        layout.addWidget(self.image_label)
        
        # Instructions
        instruction_label = QLabel("Click on the object to track")
        layout.addWidget(instruction_label)
        
        self.setLayout(layout)

    def display_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        
        # Store scale factors for coordinate conversion
        self.scale_x = w / scaled_pixmap.width()
        self.scale_y = h / scaled_pixmap.height()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Convert coordinates back to original image space
            x = event.pos().x() * self.scale_x
            y = event.pos().y() * self.scale_y
            self.point_selected.emit(x, y)
            self.close()

class SAMProcessor:
    def __init__(self):
        # Initialize SAM model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize SAM2 model
        from sam2.build_sam import build_sam2_video_predictor

        sam2_checkpoint = "/home/cyan/code/universal_manipulation_interface/umi/real_world/sam_ckpt/sam2.1_hiera_large.pt"
        model_cfg = "/home/cyan/code/universal_manipulation_interface/umi/real_world/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        
        # Visualization parameters
        self.mask_alpha = 0.3
        self.mask_color = np.array([200, 0, 200])

        self.tracking_point = None  # Single point to store tracking location

    def get_mask_center(self, mask):
        """Calculate center of mass of the mask"""
        if mask is None:
            return None
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        return [center_x, center_y]

    def set_point(self, x, y):
        """Set tracking point"""
        self.tracking_point = [x, y]


    def process_frames(self, frames):
        """Process frames using current tracking point"""
        inference_state = self.predictor.init_state()
        self.predictor.reset_state(inference_state)

        # Process only if we have a tracking point
        if self.tracking_point is not None:
            points = np.array([self.tracking_point], dtype=np.float32)
            labels = np.array([1], np.int32)
            
            # Add point and check if successful
            inference_state, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=0,
                points=points,
                labels=labels
            )
            
            # Check if point was added successfully
            if len(obj_ids) == 0:
                print("Failed to add tracking point")
                return frames

            # Generate masks for all frames
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state, start_frame_idx=0):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                video_segments[out_frame_idx] = mask

            # Process frames and update tracking point
            processed_frames = frames.copy()
            last_mask = None
            for frame_idx, mask in video_segments.items():
                last_mask = mask
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * self.mask_color.reshape(1, 1, -1)
                blended_frame = (1 - self.mask_alpha) * frames[frame_idx] + self.mask_alpha * mask_image
                processed_frames[frame_idx] = np.clip(blended_frame, 0, 255).astype(np.uint8)

            # Update tracking point using the last frame's mask
            if last_mask is not None:
                self.tracking_point = self.get_mask_center(last_mask)

            return processed_frames
        return frames

class MultiUvcCamera:
    def __init__(self,
            # v4l2 device file path
            # e.g. /dev/video0
            # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
            dev_video_paths: List[str],
            shm_manager: Optional[SharedMemoryManager]=None,
            resolution=(1280,720),
            capture_fps=60,
            put_fps=None,
            put_downsample=True,
            get_max_k=30,
            receive_latency=0.0,
            cap_buffer_size=1,
            transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
            verbose=False
        ):
        super().__init__()

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        n_cameras = len(dev_video_paths)

        resolution = repeat_to_list(
            resolution, n_cameras, tuple)
        capture_fps = repeat_to_list(
            capture_fps, n_cameras, (int, float))
        cap_buffer_size = repeat_to_list(
            cap_buffer_size, n_cameras, int)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)
        video_recorder = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)
        
        cameras = dict()
        for i, path in enumerate(dev_video_paths):
            cameras[path] = UvcCamera(
                shm_manager=shm_manager,
                dev_video_path=path,
                resolution=resolution[i],
                capture_fps=capture_fps[i],
                put_fps=put_fps,
                put_downsample=put_downsample,
                get_max_k=get_max_k,
                receive_latency=receive_latency,
                cap_buffer_size=cap_buffer_size[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose
            )

        self.cameras = cameras
        self.shm_manager = shm_manager

        self.sam_processor = SAMProcessor()
        self.app = QApplication.instance() or QApplication([])

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def get_initial_point(self, first_frame):
        """Show GUI to get initial point"""
        annotator = PointAnnotator(first_frame)
        annotator.point_selected.connect(self.sam_processor.set_point)
        annotator.show()
        self.app.exec_()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()

        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            
            # If no tracking point, get initial annotation
            if self.sam_processor.tracking_point is None:
                self.get_initial_point(this_out['rgb'][0])
            
            # Process frames with SAM
            processed_frames = self.sam_processor.process_frames(
                frames=this_out['rgb']
            )
            this_out['rgb'] = processed_frames
                
            out[i] = this_out
            
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [copy.deepcopy(x) for _ in range(n)]
    assert len(x) == n
    return x
