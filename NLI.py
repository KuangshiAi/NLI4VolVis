from icecream import install
install()
ic.configureOutput(includeContext=True) #type: ignore

import glob
import json
import os
import torchvision.transforms
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from gaussian_renderer import render_fn_dict
from scene import GaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import Camera, JSON_to_camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov,ThetaPhi2xyz,fov2focal
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform
from pyquaternion import Quaternion
import cv2
import socket
import threading
from LLM_agent import process_user_query, call_llm
from open_clip import create_model_and_transforms
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import pygame
from scene.ip2p import InstructPix2Pix
from PIL import Image
import torchvision.transforms as transforms
import shlex
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import ast
import datetime
import collections

MAX_HISTORY_SIZE = 30  # Maximal messages to keep in history

def manage_conversation_history(history, new_message):
    """
    Manage the conversation history by appending a new message and ensuring
    it does not exceed MAX_HISTORY_SIZE.
    """
    history.append(new_message)
    if len(history) > MAX_HISTORY_SIZE:
        history.pop(0)  # Remove the oldest message


def screen_to_arcball(p:np.ndarray):
    dist = np.dot(p, p)
    if dist < 1.:
        return np.array([*p, np.sqrt(1.-dist)])
    else:
        return np.array([*normalize_vec(p), 0.])

def normalize_vec(v: np.ndarray):
    if v is None:
        print("None")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.all(norm == np.zeros_like(norm)):
        return np.zeros_like(v)
    else:
        return v/norm

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def load_ckpts_paths(source_dir, stylize_name=None):
    TFs_folders = sorted(glob.glob(f"{source_dir}/TF*"))
    TFs_names = sorted([os.path.basename(folder) for folder in TFs_folders])

    ckpts_transforms = {}
    for idx, TF_folder in enumerate(TFs_folders):
        one_TF_json = {'path': None, 'palette':None, 'transform': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        if stylize_name and os.path.exists(os.path.join(TF_folder, "neilf", stylize_name, 'time.txt')):
            ckpt_dir = os.path.join(TF_folder, "neilf", stylize_name)
        else:
            ckpt_dir = os.path.join(TF_folder, "neilf", "point_cloud")
        max_iters = searchForMaxIteration(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "point_cloud.ply")
        palette_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "palette_colors_chkpnt.pth")
        one_TF_json['path'] = ckpt_path
        one_TF_json['palette'] = palette_path
        ckpts_transforms[TFs_names[idx]] = one_TF_json

    return ckpts_transforms

def scene_composition(scene_dict: dict, dataset: ModelParams):
    gaussians_list = []
    for scene in scene_dict:
        gaussians = GaussianModel(dataset.sh_degree, render_type="phong")
        print("Compose scene from GS path:", scene_dict[scene]["path"])
        gaussians.my_load_ply(scene_dict[scene]["path"], quantised=True, half_float=True)
        
        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda").reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)

    gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    return gaussians_composite

class ArcBallCamera:
    def __init__(self, W, H, fovy=60, near=0.1, far=10, rot=None, translate=None, center=None):
        self.W = W
        self.H = H
        if translate is None:
            self.radius = 1
            self.original_radius = 1
        else:
            self.radius = np.linalg.norm(translate)
            self.original_radius = np.linalg.norm(translate)
            
        # self.radius *= 2
        self.radius *= 2
        self.fovy = fovy  # in degree
        self.near = near
        self.far = far

        if center is None:
            self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        else:
            self.center = center
        
        self.original_center = self.center

        if rot is None:
            self.rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))  # looking back to z axis
            self.original_rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        else:
            self.rot = R.from_matrix(rot)
            self.original_rot = R.from_matrix(rot)

        # self.up = np.array([0, -1, 0], dtype=np.float32)  # need to be normalized!
        self.up = -self.rot.as_matrix()[:3, 1]

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)
    
    def reset_view(self):
        self.rot = self.original_rot
        self.radius = self.original_radius
        self.radius *= 2
        self.center = np.array([0, 0, 0], dtype=np.float32)

    def set_view(self, rot, radius):
        self.rot = rot
        self.radius = radius
        self.radius *= 2

    def orbit(self, lastX, lastY, X, Y):
        def vec_angle(v0: np.ndarray, v1: np.ndarray):
            return np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1., 1.))
        ball_start = screen_to_arcball(np.array([lastX+1e-6, lastY+1e-6]))
        ball_curr = screen_to_arcball(np.array([X, Y]))
        rot_radians = vec_angle(ball_start, ball_curr)
        rot_axis = normalize_vec(np.cross(ball_start, ball_curr))
        q = Quaternion(axis=rot_axis, radians=rot_radians)
        self.rot = self.rot * R.from_matrix(q.inverse.rotation_matrix)
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

def replace_color_to_contrast(color):
    return (1 - color) * 0.7

class GUI:
    def __init__(self, H, W, fovy, c2w, center, render_fn, render_kwargs, TFnums, args,
                 mode="phong", debug=True):
        """
        If the image is hdr, set use_hdr2ldr = True for LDR visualization. [0, 1]
        If the image is hdr, set use_hdr2ldr = False, the range of the image is not [0,1].
        """
        self.ctrlW = 550 #475
        self.chatW = 510
        self.commandW = 300
        self.widget_indent = 75
        self.widget_top = 150
        self.imgW = W
        self.imgH = H
        self.debug = debug
        rot = c2w[:3, :3]
        translate = c2w[:3, 3] - center
        self.TFnums = TFnums
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs
        self.original_palette_colors = [self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color for TFidx in range(TFnums)]
        
        # self.original_rot = rot
        # self.original_translate = translate
        
        # self.cam = OrbitCamera(self.imgW, self.imgH, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)
        self.cam = ArcBallCamera(self.imgW, self.imgH, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)

        self.render_buffer = np.zeros((self.imgW, self.imgH, 3), dtype=np.float32)
        self.resize_fn = torchvision.transforms.Resize((self.imgH, self.imgW), antialias=True)
        self.downsample = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.prev_mouseX = None
        self.prev_mouseY = None
        self.rotating = False
        
        self.light_elevation = 0
        self.light_angle = 180
        self.useHeadlight = True
        
        self.menu = None
        self.mode = None

        # Chat conversation histories
        self.conversation_history_parser = collections.deque(maxlen=MAX_HISTORY_SIZE)
        self.conversation_history_controller = collections.deque(maxlen=MAX_HISTORY_SIZE)
        
        # A text buffer to display the conversation
        self.chat_log = ""

        self.args = args

        # Optionally store references to agent's embeddings
        if 'deepseek' in args.llm_name.lower():
            client = OpenAI(api_key=args.api_key[args.llm_name], base_url="https://api.deepseek.com")
        elif 'gpt' in args.llm_name.lower():
            client = OpenAI(api_key=args.api_key[args.llm_name])
        else:
            client = OpenAI(api_key=args.api_key[args.llm_name], base_url="https://api.llama-api.com")

        # Load CLIP model
        clip_model, preprocess_train, preprocess_val = create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_model.eval()

        # Load TF embeddings
        tf_embeddings = {}
        for folder_name in os.listdir(args.image_path):
            if folder_name.startswith("TF"):
                embedding_path = os.path.join(args.image_path, folder_name, args.embedding_name)
                if os.path.exists(embedding_path):
                    tf_embeddings[folder_name] = np.load(embedding_path)
        self.tf_embeddings = tf_embeddings
        self.llm_client = client
        self.clip_model = clip_model
        self.query_step = 0
        self.img_path = args.image_path
        self.llm_name = args.llm_name

        # variables about audio recording
        self.audio_client = OpenAI(api_key=args.api_key['openai_audio'])
        self.is_recording = False  # Recording state flag
        self.audio_data = []
        self.fs = 44100
        self.recording_thread = None
        pygame.mixer.init()
        self.audio_mute = False

        # Stylization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ip2p = InstructPix2Pix(device=device, ip2p_use_full_precision=False)

        # Legend
        self.legend_dict = {}

        self.default_fov = self.cam.fovy

        self.freeze_view = False

        self.step()
        self.mode = "phong"
        dpg.create_context()
        
        self.setup_font_theme()
        # dpg.bind_item_font(your_item, default_font)
        
        light_theme = create_theme_imgui_light()
        dpg.bind_theme(light_theme)
        self.register_dpg()
        # Create the chat window
        self.register_chat_window()
        self.initialize_visualization()
        self.start_socket_server()

    def __del__(self):
        pygame.mixer.music.stop() 
        dpg.destroy_context()

    def get_status(self):
        status = {
            "mode": self.mode,
            "field_of_view": self.cam.fovy,
            "background_color": self.render_kwargs["bg_color"].tolist(),
            "opacity_factors": [
                opacity.opacity_factor.item() if isinstance(opacity.opacity_factor, torch.Tensor) else opacity.opacity_factor
                for opacity in self.render_kwargs["dict_params"]["opacity_factors"]
            ],
            "palette_colors": [
                color.palette_color.tolist()
                for color in self.render_kwargs["dict_params"]["palette_colors"]
            ],
            "light": {
                "angle": self.light_angle,
                "elevation": self.light_elevation,
                "ambient": self.render_kwargs["dict_params"]["light_transform"].ambient_multi 
                        if not isinstance(self.render_kwargs["dict_params"]["light_transform"].ambient_multi, torch.Tensor)
                        else self.render_kwargs["dict_params"]["light_transform"].ambient_multi.item(),
                "diffuse": self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi
                        if not isinstance(self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi, torch.Tensor)
                        else self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi.item(),
                "specular": self.render_kwargs["dict_params"]["light_transform"].specular_multi
                            if not isinstance(self.render_kwargs["dict_params"]["light_transform"].specular_multi, torch.Tensor)
                            else self.render_kwargs["dict_params"]["light_transform"].specular_multi.item(),
                "shininess": self.render_kwargs["dict_params"]["light_transform"].shininess_multi
                            if not isinstance(self.render_kwargs["dict_params"]["light_transform"].shininess_multi, torch.Tensor)
                            else self.render_kwargs["dict_params"]["light_transform"].shininess_multi.item(),
            },
            "freeze_view": self.freeze_view,
            "legend": self.legend_dict,
        }
        return status

    def process_message(self, message, conn):
        def can_convert_to_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
            
        # Log the received command
        self.append_command_log(message)

        # Parse and handle incoming messages here
        if message.startswith("set_opacity"):
            _, tf_index, value = message.split()
            tf_index = int(tf_index)
            value = float(value)
            slider_tag = f"_slider_TF{tf_index}"
            if dpg.does_item_exist(slider_tag):
                dpg.set_value(slider_tag, value)
                with torch.no_grad():
                    self.render_kwargs["dict_params"]["opacity_factors"][tf_index].opacity_factor = torch.tensor(
                        value, dtype=torch.float32, device="cuda"
                    )
                self.need_update = True  # Notify the GUI to refresh the visualization

        elif message.startswith("set_color"):
            _, tf_index, r, g, b = message.split()
            tf_index = int(tf_index)
            color_tag = f"_color_TF{tf_index}"
            color_value = (int(r), int(g), int(b), 255)
            if dpg.does_item_exist(color_tag):
                dpg.set_value(color_tag, color_value)
                with torch.no_grad():
                    self.render_kwargs["dict_params"]["palette_colors"][tf_index].palette_color = torch.tensor(
                        [int(r) / 255, int(g) / 255, int(b) / 255], dtype=torch.float32, device="cuda"
                    )
                self.need_update = True

        elif message.startswith("set_light"):
            parts = message.split()
            if len(parts) == 3:
                if can_convert_to_float(parts[2]):
                    param, value = parts[1], float(parts[2])
                else:
                    param = parts[1]
                if param == "angle" or param == "elevation":
                    if param == "angle":
                        dpg.set_value("_slider_light_angle", value)
                        self.light_angle = value
                    elif param == "elevation":
                        dpg.set_value("_slider_light_elevation", value)
                        self.light_elevation = value
                    self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(
                        self.light_angle, self.light_elevation
                    )
                elif param == "ambient":
                    dpg.set_value("_slider_ambient_multi", value)
                    self.render_kwargs["dict_params"]["light_transform"].ambient_multi = torch.tensor(
                        value, dtype=torch.float32, device="cuda"
                    )
                elif param == "diffuse":
                    dpg.set_value("_slider_light_intensity_multi", value)
                    self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi = torch.tensor(
                        value, dtype=torch.float32, device="cuda"
                    )
                elif param == "specular":
                    dpg.set_value("_slider_specular_multi", value)
                    self.render_kwargs["dict_params"]["light_transform"].specular_multi = torch.tensor(
                        value, dtype=torch.float32, device="cuda"
                    )
                elif param == "shininess":
                    dpg.set_value("_slider_shininess_multi", value)
                    self.render_kwargs["dict_params"]["light_transform"].shininess_multi = torch.tensor(
                        value, dtype=torch.float32, device="cuda"
                    )
                elif parts[1] == "headlight":
                    value = parts[2].lower() == "true"
                    dpg.set_value("_checkbox_headlight", value)
                    self.useHeadlight = value
                    self.render_kwargs["dict_params"]["light_transform"].useHeadLight = self.useHeadlight
                else:
                    raise ValueError(f"Invalid message: {message}")
                self.need_update = True
            else:
                raise ValueError(f"Invalid message: {message}")
            
        elif message.startswith("set_mode"):
            parts = message.split()
            if len(parts) == 2:
                _, mode = parts
                if mode in self.menu_map:  # Validate if the mode exists in menu_map
                    self.mode = mode  # Update the internal mode
                    gui_mode = self.menu_map[mode]  # Get the user-facing name of the mode
                    dpg.set_value("_log_infer_time", f"Mode set to {gui_mode}")
                    dpg.set_value("_combo_mode", gui_mode)  # Update the combo box value
                    self.need_update = True
                    self.step()  # Refresh visualization
                else:
                    print(f"Invalid mode: {mode}. Available modes: {list(self.menu_map.keys())}")
            else:
                print("Invalid set_mode command format. Use: set_mode <mode>")

        elif message.startswith("set_fov"):
            _, fov = message.split()
            fov = int(fov)
            if 1 <= fov <= 120:
                dpg.set_value("_slider_fovy", fov)
                self.cam.fovy = fov
                self.need_update = True
            else:
                print("Field of View must be between 1 and 120 degrees.")

        elif message.startswith("set_background"):
            _, r, g, b = message.split()
            bg_color = [int(r) / 255, int(g) / 255, int(b) / 255]
            dpg.set_value("_color_edit_background", (int(r), int(g), int(b), 255))
            self.render_kwargs["bg_color"] = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            self.need_update = True

        elif message.startswith("get_status"):
            #print("Current Status:", self.get_status())
            status_json = json.dumps(self.get_status())
            conn.sendall(f"Current Status: {status_json}\n".encode("utf-8"))
        
        elif message.startswith("reset_view"):
            file_path = os.path.join(self.img_path, "initial_view.txt")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                    # Find index for the "Radius:" line.
                    radius_index = None
                    for i, line in enumerate(lines):
                        if line.startswith("Radius:"):
                            radius_index = i
                            break
                    if radius_index is None or radius_index < 2:
                        print("Initial view file format is incorrect.")
                    else:
                        # Join the lines that contain the rotation matrix (from index 1 to radius_index).
                        rot_str = " ".join(line.strip() for line in lines[1:radius_index])
                        rot_matrix = np.array(ast.literal_eval(rot_str))
                        rot_obj = R.from_matrix(rot_matrix)
                        
                        # Parse the radius.
                        radius_line = lines[radius_index].strip()  # e.g., "Radius: 3.305785123966942"
                        radius = float(radius_line.split("Radius:")[1].strip())
                        
                        # Parse the center from the next line.
                        center_line = lines[radius_index+1].strip()  # e.g., "Center: [ 0.06522223, -0.02958885,  0.00098899]"
                        center = np.array(ast.literal_eval(center_line.split("Center:")[1].strip()))
                        
                        # Set the camera parameters.
                        self.cam.rot = rot_obj
                        self.cam.radius = radius
                        self.cam.center = center
                        
                        print("Loaded initial view from", file_path)
                except Exception as e:
                    print("Error loading initial view:", e)
                    self.cam.reset_view()
            else:
                print("Initial view file not found:", file_path)
                self.cam.reset_view()
            if self.freeze_view:
                dpg.configure_item("_freeze_view_button", label="Freeze View")
                print("View unfrozen.")
                self.freeze_view = False
            self.need_update = True
        
        elif message.startswith("save_image"):
            rendered_img = self.save_rgba_buffer
            rendered_img = (rendered_img*255).astype(np.uint8)[...,[2,1,0,3]]
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the image with the timestamp in the filename
            filename = f'./screenshots/user0/rendered_img_{timestamp}.png'
            cv2.imwrite(filename, rendered_img)
            print("Image Saved")
        
        elif message.startswith("reset_color_opacity"):
            with torch.no_grad():
                for TFidx in range(self.TFnums):
                    self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                    self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                    color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                    dpg.set_value(f"_slider_TF{TFidx}", 1)
                    dpg.set_value(f"_color_TF{TFidx}", tuple(color_value))
            self.legend_dict = {}
            self.need_update = True
        elif message.startswith("set_view"):
            _, tf_number, frame_number = message.split()
            tf_number = int(tf_number)
            frame_number = int(frame_number)
            TFs_folders = sorted(glob.glob(f"{args.image_path}/TF*"))
            view_config_file = f"{TFs_folders[tf_number]}/transforms_train.json"
            view_dict = load_json_config(view_config_file)
            all_views = view_dict["frames"]
            c2w = np.array(all_views[frame_number]["transform_matrix"]).reshape(4, 4)
            c2w /= 2
            c2w[:3, 1:3] *= -1
            # Extract rotation matrix (top-left 3x3) and translation vector (top 3 elements of last column)
            rotation_matrix = c2w[:3, :3]
            translation = c2w[:3, 3]

            # Compute radius
            radius = np.linalg.norm(translation)

            # Convert rotation matrix to scipy Rotation object
            rot = R.from_matrix(rotation_matrix)

            self.cam.set_view(rot, radius)

            self.need_update = True
        elif message.startswith("stylize"):
            parts = shlex.split(message)
            tf_numbers = parts[1]
            prompt = parts[2]
            dpg.configure_item("_freeze_view_button", label="Unfreeze View")
            self.freeze_view = True
            self.step()
            if tf_numbers == "whole":
                #tf_numbers = list(range(self.TFnums))
                # Run the IP2P process in a separate thread so as not to block the GUI.
                threading.Thread(target=self.process_ip2p_prompt, args=(prompt,), daemon=True).start()
            else:
                tf_numbers = [int(x) for x in tf_numbers.split("&")]
                threading.Thread(target=self.process_ip2p_prompt, args=(prompt, tf_numbers), daemon=True).start()


        elif message.startswith("legend add"):
            # Use shlex.split to handle quotes in the text label.
            parts = shlex.split(message)
            if len(parts) != 6:
                print("Invalid legend add command format. Use: legend add <text> <r> <g> <b>")
            else:
                # parts[0] is "legend", parts[1] is "add", parts[2] is the text label,
                # and parts[3], parts[4], parts[5] are the color components.
                label = parts[2]
                try:
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                except ValueError:
                    print("Color values must be integers.")
                    return
                # Initialize the legend dictionary if it doesn't exist.
                if not hasattr(self, "legend_dict"):
                    self.legend_dict = {}
                self.legend_dict[label] = [r, g, b]
                print(f"Added legend entry: {label} -> {[r, g, b]}")
                #self.append_command_log(f"Added legend entry: {label} -> {[r, g, b]}")
                self.need_update = True

        elif message.startswith("legend delete"):
            parts = shlex.split(message)
            if len(parts) != 3:
                print("Invalid legend delete command format. Use: legend delete <text>")
            else:
                label = parts[2]
                if hasattr(self, "legend_dict") and label in self.legend_dict:
                    del self.legend_dict[label]
                    print(f"Deleted legend entry: {label}")
                    #self.append_command_log(f"Deleted legend entry: {label}")
                    self.need_update = True
                else:
                    print(f"Legend entry '{label}' not found.")
        
        elif message.startswith("start_tour"):
            if self.initialize_visualization():
                print("Tour started.")
                dpg.configure_item("_freeze_view_button", label="Unfreeze View")
                print("View frozen.")
                self.freeze_view = True
            self.need_update = True

        else:
            print(f"Unknown command: {message}")

        # Force the render to refresh
        self.step()
    
    def initialize_visualization(self):
        with torch.no_grad():
            for TFidx in range(self.TFnums):
                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                #color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                color_tensor = self.original_palette_colors[TFidx].detach().cpu()
                color_array = np.nan_to_num(color_tensor.numpy(), nan=0.0)  # replace NaN with 0.0
                color_value = [int(x * 255) for x in color_array]
                dpg.set_value(f"_slider_TF{TFidx}", 1)
                dpg.set_value(f"_color_TF{TFidx}", tuple(color_value))
        file_path = os.path.join(self.img_path, "initial_view.txt")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                # Find index for the "Radius:" line.
                radius_index = None
                for i, line in enumerate(lines):
                    if line.startswith("Radius:"):
                        radius_index = i
                        break
                if radius_index is None or radius_index < 2:
                    print("Initial view file format is incorrect.")
                else:
                    # Join the lines that contain the rotation matrix (from index 1 to radius_index).
                    rot_str = " ".join(line.strip() for line in lines[1:radius_index])
                    rot_matrix = np.array(ast.literal_eval(rot_str))
                    rot_obj = R.from_matrix(rot_matrix)
                    
                    # Parse the radius.
                    radius_line = lines[radius_index].strip()  # e.g., "Radius: 3.305785123966942"
                    radius = float(radius_line.split("Radius:")[1].strip())
                    
                    # Parse the center from the next line.
                    center_line = lines[radius_index+1].strip()  # e.g., "Center: [ 0.06522223, -0.02958885,  0.00098899]"
                    center = np.array(ast.literal_eval(center_line.split("Center:")[1].strip()))
                    
                    # Set the camera parameters.
                    self.cam.rot = rot_obj
                    self.cam.radius = radius
                    self.cam.center = center
                    
                    print("Loaded initial view from", file_path)
            except Exception as e:
                print("Error loading initial view:", e)
                self.need_update = True
                return False
        else:
            print("Initial view file not found:", file_path)
            self.need_update = True
            return False
        self.need_update = True
        return True

    def start_socket_server(self, host="127.0.0.1", port=65432):
        def handle_client_connection(conn):
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message = data.decode("utf-8")
                    print(f"Received: {message}")
                    self.process_message(message, conn)

        def server_thread():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                s.listen()
                print(f"Socket server listening on {host}:{port}")
                while True:
                    conn, addr = s.accept()
                    print(f"Connected by {addr}")
                    threading.Thread(target=handle_client_connection, args=(conn,)).start()

        threading.Thread(target=server_thread, daemon=True).start()

    
    def setup_font_theme(self):
        with dpg.font_registry():
            self.default_font = dpg.add_font("./assets/font/Helvetica.ttf", 16)
            with dpg.font("./assets/font/Helvetica.ttf", 20) as self.chat_font:
                # add the default font range
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)

                # helper to add range of characters
                #    Options:
                #        mvFontRangeHint_Japanese
                #        mvFontRangeHint_Korean
                #        mvFontRangeHint_Chinese_Full
                #        mvFontRangeHint_Chinese_Simplified_Common
                #        mvFontRangeHint_Cyrillic
                #        mvFontRangeHint_Thai
                #        mvFontRangeHint_Vietnamese
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
            
            with dpg.font("./assets/font/Helvetica.ttf", 22) as self.bubble_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (161, 238, 189)) #(139, 205, 162)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (174, 255, 204))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (205, 250, 219)) #(174, 255, 203)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_font(self.default_font)
        self.theme_button = theme_button
        with dpg.theme() as self.child_window_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255))

    def get_buffer(self, render_results, mode=None):
        if render_results is None or mode is None:
            output = torch.ones(self.imgH, self.imgW, 3, dtype=torch.float32, device='cuda').detach().cpu().numpy()
        else:
            output = render_results[mode]
            
            if mode == "depth":
                output = (output - output.min()) / (output.max() - output.min())
            elif mode == "num_contrib":
                output = output.clamp_max(1000) / 1000

            if len(output.shape) == 2:
                output = output[None]
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            if "normal" in mode:
                opacity = render_results["opacity"]
                output = output * 0.5 + 0.5 * opacity
                output = output + (1 - opacity)
            elif mode in ["diffuse_term", "specular_term", "ambient_term"]:
                opacity = render_results["opacity"]
                output = output + (1 - opacity)
            if (self.imgH, self.imgW) != tuple(output.shape[1:]):
                output = self.resize_fn(output)

            output = output.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        return output

    def get_rgba_buffer(self, render_results, mode=None):
        if render_results is None or mode is None:
            output = torch.ones(self.imgH, self.imgW, 3, dtype=torch.float32, device='cuda').detach().cpu().numpy()
        else:
            output = render_results[mode]
            
            if mode == "depth":
                output = (output - output.min()) / (output.max() - output.min())
            elif mode == "num_contrib":
                output = output.clamp_max(1000) / 1000

            if len(output.shape) == 2:
                output = output[None]
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            if "normal" in mode:
                opacity = render_results["opacity"]
                output = output * 0.5 + 0.5 * opacity
                output = output + (1 - opacity)
            elif mode in ["diffuse_term", "specular_term", "ambient_term"]:
                opacity = render_results["opacity"]
                output = output + (1 - opacity)
            if (self.imgH, self.imgW) != tuple(output.shape[1:]):
                output = self.resize_fn(output)

            output = output.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        # If output only has 3 channels, add an alpha channel set to 1 (opaque)
        if output.shape[-1] == 3:
            alpha = render_results["opacity"].permute(1, 2, 0).detach().cpu().numpy().astype(output.dtype)
            output = np.concatenate([output, alpha], axis=-1)
        return output
    
    def overlay_legend(self, render_buffer, legend_dict, position=(10, 10), square_size=60, padding=15):
        #60, 15
        #40, 10
        #20, 5
        """
        Overlays a legend on the input image.
        
        Parameters:
            render_buffer (np.ndarray): Image array with values in [0,1].
            legend_dict (dict): Dictionary with keys as labels and values as [R, G, B] colors (0-255).
            position (tuple): (x, y) position for the top-left corner of the legend.
            square_size (int): Size of the color square.
            padding (int): Padding between legend items.
            
        Returns:
            np.ndarray: The image with the legend overlay, normalized to [0,1].
        """
        # If legend_dict is empty, return the original render_buffer
        if not legend_dict:
            return render_buffer
        # Convert the render_buffer (float in [0,1]) to a PIL Image in uint8 format.
        img = Image.fromarray((render_buffer * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
        # Try loading a TrueType font; if unavailable, use the default font.
        try:
            font = ImageFont.truetype("./assets/font/Helvetica.ttf", 50) #22, 35, 50
        except IOError:
            font = ImageFont.load_default()

        x, y = position
        for label, rgb in legend_dict.items():
            # Draw a filled rectangle (the colored square)
            draw.rectangle([x, y, x + square_size, y + square_size], fill=tuple(rgb))
            # Draw the label text to the right of the square (text in black)
            draw.text((x + square_size + padding, y), label, fill=(0, 0, 0), font=font)
            # Move y for the next legend item
            y += square_size + padding

        # Convert the PIL image back to a NumPy array normalized to [0,1]
        annotated_img = np.array(img).astype(np.float32) / 255.0
        return annotated_img

    @property
    def custom_cam(self):
        w2c = self.cam.view
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        down = self.downsample
        H, W = self.imgH // down, self.imgW // down
        fovy = self.cam.fovy * np.pi / 180
        fovx = fovy * W / H
        custom_cam = Camera(colmap_id=0, R=R, T=-T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0)
        return custom_cam

    @torch.no_grad()
    def render(self):
        if getattr(self, "need_update", False):
            self.step()  # update texture from the scene render, etc.
            self.need_update = False  # reset flag after updating
        dpg.render_dearpygui_frame()


    def step(self):
        self.start.record()
        render_pkg = self.render_fn(viewpoint_camera=self.custom_cam, **self.render_kwargs)
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)

        buffer1 = self.get_buffer(render_pkg, self.mode)
        # Overlay legend on the render buffer
        self.render_buffer = self.overlay_legend(buffer1, self.legend_dict)

        buffer2 = self.get_rgba_buffer(render_pkg, self.mode)
        self.save_rgba_buffer = buffer2
        

        if t == 0:
            fps = 0
        else:
            fps = int(1000 / t)

        if self.menu is None:
            #* Forgive me for this ugly fix for menu
            self.menu_map = {"phong": "Blinn-Phong", "normal": "Normal", "diffuse_term": "Diffuse",
                             "specular_term": "Specular", "ambient_term": "Ambient"}
            self.inv_menu_map = {v: k for k, v in self.menu_map.items()}
            self.menu = [self.menu_map[k] for k, v in render_pkg.items() if
                         k not in ["pseudo_normal","render", "num_contrib", "surface_xyz", "diffuse_factor","depth", "shininess", "ambient_factor", "specular_factor", "offset_color", "opacity"] and isinstance(v, torch.Tensor) and np.array(v.shape).prod() % (self.imgH * self.imgW) == 0]
            self.menu = ["Blinn-Phong", "Ambient", "Diffuse", "Specular", "Normal"]
            
        else:
            dpg.set_value("_log_infer_time", f'{t:.4f} ms ({fps} FPS)')
            dpg.set_value("_texture", self.render_buffer)
        torch.cuda.empty_cache()
    
    def add_oneTFSlider(self, TFidx):
        def callback_TF_slider(sender, app_data):
            TFidx = int(sender.replace("_slider_TF", ""))
            with torch.no_grad():
                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(app_data, dtype=torch.float32, device="cuda")
            self.need_update = True
        
        def callback_TF_color_edit(sender, app_data):
            TFidx = int(sender.replace("_color_TF", ""))
            with torch.no_grad():
                self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = torch.tensor(app_data[:3], dtype=torch.float32, device="cuda")
            self.need_update = True
        
        slider_tag = "_slider_TF" + str(TFidx)
        color_tag = "_color_TF" + str(TFidx)
        defualt_color = self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color.detach().cpu().numpy()
        defualt_color = (defualt_color * 255).astype(np.uint8).tolist()
        
        # indent = self.widget_indent if TFidx == 0 else 0
        indent = 0
        slider_width = (self.ctrlW-10)//(self.TFnums) # leave some space (10 pixels) at right
      
        with dpg.group():
            dpg.add_text(f"TF{TFidx}",indent=indent+slider_width//4 if self.TFnums <11 else indent+slider_width//(self.TFnums/2))
            dpg.add_slider_float(
                tag=slider_tag,
                label='',
                default_value=0,
                min_value=0,
                max_value=3.0,
                height=300,
                # format="",
                callback=callback_TF_slider,
                vertical=True,
                width=slider_width, 
                indent=indent
            )
            dpg.add_color_edit(tag=color_tag, default_value=defualt_color, callback=callback_TF_color_edit,
                               no_inputs=True, no_label=True, no_alpha=True, indent=indent+slider_width//4 if self.TFnums <11 else indent+slider_width//(self.TFnums/2))

    def register_chat_window(self):
        with dpg.theme() as self.user_bubble_theme:
            with dpg.theme_component(dpg.mvAll):
                # user bubble color
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (210, 255, 210, 255))  
                # corner rounding
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                # no-spacing
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 5, 5)

        with dpg.theme() as self.assistant_bubble_theme:
            with dpg.theme_component(dpg.mvAll):
                # assistant bubble color
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (225, 225, 235, 255))
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                # no-spacing
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 5, 5)
        
        with dpg.theme() as self.system_bubble_theme:
            with dpg.theme_component(dpg.mvAll):
                # system bubble color - e.g. light grey
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 204, 255))
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 5, 5)

        with dpg.window(label="Chat with LLM Agent", tag="_chat_window",
                        width=self.chatW, height=self.imgH, pos=(self.imgW + self.ctrlW, 0),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=False):
            # Title
            dpg.add_text("Conversation")
            dpg.bind_item_font(dpg.last_item(), self.chat_font)  # Apply larger font

            with dpg.child_window(tag="chat_scroll_window", width=self.chatW, height=580, no_scrollbar=True):
                dpg.add_child_window(tag="chat_history",width=self.chatW-25, height=580)
                dpg.bind_item_theme(dpg.last_item(), self.child_window_theme)

            dpg.add_text("Select an LLM")
            dpg.bind_item_font(dpg.last_item(), self.chat_font)
            dpg.add_combo(
                ["gpt-4o", "deepseek-chat", "llama3.2-90b-vision"],
                tag="llm_selector",
                default_value=self.llm_name,
                callback=self.on_llm_selected
            )
            dpg.bind_item_font(dpg.last_item(), self.chat_font)

            # User input label
            dpg.add_text("Your Message")
            dpg.bind_item_font(dpg.last_item(), self.chat_font)  # Apply larger font

            # User input + Send button
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="chat_input",
                    default_value="",
                    width=400,
                    height=40,
                    multiline=False,
                    on_enter=True,  # Allow pressing "Enter" to send
                    callback=self.on_chat_send_clicked
                )
                dpg.bind_item_font(dpg.last_item(), self.chat_font)

                dpg.add_button(
                    label="Send",
                    width=70,
                    callback=self.on_chat_send_clicked
                )
                dpg.bind_item_font(dpg.last_item(), self.chat_font)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Tap to Speak",  # Unicode for "microphone" icon in FontAwesome
                    width=140,
                    tag="audio_record_button",
                    callback=self.on_audio_record_clicked
                )
                dpg.bind_item_font(dpg.last_item(), self.chat_font)
                dpg.bind_item_theme("audio_record_button", self.theme_button)
                dpg.add_button(
                    label="Unmute",  # Start with speaker icon (unmute)
                    width=100,
                    tag="mute_toggle_button",
                    callback=self.on_mute_toggle
                )
                dpg.bind_item_font(dpg.last_item(), self.chat_font)
            
            # # User input label
            # dpg.add_text("Your Stylization Prompt")
            # dpg.bind_item_font(dpg.last_item(), self.chat_font)  # Apply larger font

            # with dpg.group(horizontal=True):
            #     dpg.add_input_text(tag="ip2p_input",          
            #         default_value="",
            #         width=400,
            #         height=40,
            #         multiline=False,
            #         on_enter=True,  # Allow pressing "Enter" to send
            #         callback=self.on_ip2p_prompt)
            #     dpg.bind_item_font(dpg.last_item(), self.chat_font)
            #     dpg.add_button(label="Send", tag="ip2p_button", width=70, callback=self.on_ip2p_prompt)
            #     dpg.bind_item_font(dpg.last_item(), self.chat_font)

            # Generate welcoming message from basic dataset information
            try:
                dataset_info_path = os.path.join(self.img_path, "dataset_info.txt")
                with open(dataset_info_path, "r") as f:
                    dataset_info = f.read().strip()
                self.dataset_info = dataset_info
            except Exception as e:
                dataset_info = "Dataset information not available."
            
            welcome_prompt = (
                "You are an assistant that converts user requests into GUI control commands. "
                "Given the dataset info, generate a concise welcome message that mentions the dataset name and lists the available objects. "
                "Also explain that the user can adjust properties like opacity, color, lighting, camera view, or apply image stylization using a text prompt. "
                "Sometimes class names may be unintelligible or verbose, so use your best judgment to interpret them. "
                "For example, your response could be:\n\n"
                "\"Welcome to NLI4VolVis! Your dataset 'backpack_obj' includes a bottle, box, string, and toothpaste.\n"
                "You can adjust these objects’ opacity, color, lighting, or camera view, or apply stylization using a text prompt. For example:\n"
                "1. 'Change the opacity of the bottle to 0.5'\n"
                "2. 'Set the color of the box to bright blue'\n"
                "3. 'Rotate the view to see the string from a new angle'\n"
                "How can I help you today?\""
            )

            # Set the conversation history to include the dataset info.
            conversation_history = [{"role": "user", "content": dataset_info}]

            # Call the LLM function.
            initial_message = call_llm(conversation_history, self.llm_client, system=welcome_prompt, model=self.llm_name)

            # Append the generated message to the chat window.
            self.append_chat_bubble("Assistant", initial_message)
        
        dpg.bind_item_theme("_chat_window", self.white_bg_theme)

        # Command log window (placed to the right of chat window)
        with dpg.window(label="Command Log", tag="_command_window",
                        width=self.commandW, height=400, pos=(self.imgW + self.ctrlW + self.chatW, 0),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=False):
            dpg.add_text("Agent Commands")
            dpg.bind_item_font(dpg.last_item(), self.chat_font)

            # Scrollable command log
            with dpg.child_window(tag="command_scroll_window", width=self.commandW, height=350, no_scrollbar=True):
                dpg.add_child_window(tag="command_history",width=self.commandW-25, height=350)
                dpg.bind_item_theme(dpg.last_item(), self.child_window_theme)
        
        dpg.bind_item_theme("_command_window", self.white_bg_theme)

        # Open-vocabulary Result window
        with dpg.window(label="Query Log", tag="_query_window",
                        width=self.commandW, height=400, pos=(self.imgW + self.ctrlW + self.chatW, 400),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=False):
            dpg.add_text("Open-Vocabulary Queries")
            dpg.bind_item_font(dpg.last_item(), self.chat_font)
            # Scrollable command log
            with dpg.child_window(tag="query_scroll_window", width=self.commandW, height=350, no_scrollbar=True):
                dpg.add_child_window(tag="query_history",width=self.commandW-25, height=350)
                dpg.bind_item_theme(dpg.last_item(), self.child_window_theme)
        
        dpg.bind_item_theme("_query_window", self.white_bg_theme)


    def on_mute_toggle(self, sender, app_data):
        """Toggle mute/unmute state and update the button icon."""
        self.audio_mute = not self.audio_mute
        new_label = "Mute" if self.audio_mute else "Unmute"
        dpg.configure_item("mute_toggle_button", label=new_label)
        #dpg.set_value(sender, new_label)  # Update the button label to reflect new state.
        # Optionally, if you want to immediately stop any playing audio when muting:
        if self.audio_mute:
            pygame.mixer.music.stop()
        print("Audio mute toggled. mute:", self.audio_mute)
    
    def on_freeze_view(self, sender, app_data):
        self.freeze_view = not self.freeze_view
        if self.freeze_view:
            # Save the current camera parameters.
            self.frozen_rot = self.cam.rot
            self.frozen_radius = self.cam.radius
            self.frozen_center = self.cam.center

            # Save the parameters to a file in the image path.
            file_path = os.path.join(self.img_path, "initial_view.txt")
            with open(file_path, "w") as f:
                # Convert the rotation to a matrix string.
                rot_matrix_str = np.array2string(self.frozen_rot.as_matrix(), separator=", ")
                f.write("Rotation Matrix:\n")
                f.write(rot_matrix_str + "\n")
                f.write("Radius: " + str(self.frozen_radius) + "\n")
                f.write("Center: " + np.array2string(self.frozen_center, separator=", ") + "\n")

            dpg.configure_item(sender, label="Unfreeze View")
            print("View frozen.")
        else:
            dpg.configure_item(sender, label="Freeze View")
            print("View unfrozen.")
    
    def on_ip2p_prompt(self, sender, app_data):
        prompt = dpg.get_value("ip2p_input").strip()
        self.append_chat_bubble("Stylization", prompt)
        dpg.set_value("ip2p_input", "")  # clear input
        if prompt == "":
            print("Please enter a prompt for stylization.")
            return
        # Run the IP2P process in a separate thread so as not to block the GUI.
        threading.Thread(target=self.process_ip2p_prompt, args=(prompt,), daemon=True).start()
    
    def process_ip2p_prompt(self, prompt, tf_numbers=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        guidance_scale = 18 #8
        image_guidance_scale = 0.75 #2
        diffusion_steps = 25
        lower_bound = 0.7
        upper_bound = 0.98

        if tf_numbers is not None:
            # ===== Step 1: Save original opacities =====
            ori_tf_opacity = {
                i: (opacity.opacity_factor.item() if isinstance(opacity.opacity_factor, torch.Tensor) else opacity.opacity_factor)
                for i, opacity in enumerate(self.render_kwargs["dict_params"]["opacity_factors"])
            }
            print("Original TF opacities:", ori_tf_opacity)

            # ===== Step 2: Update opacities =====
            # For each TF: if its index is in tf_numbers, set it to 1.0 (non-zero); otherwise, set to 0.0.
            for i in range(len(self.render_kwargs["dict_params"]["opacity_factors"])):
                if i in tf_numbers:
                    if ori_tf_opacity[i] == 0.0:
                        new_val = 1.0
                    else:
                        new_val = ori_tf_opacity[i]
                else:
                    new_val = 0.0
                slider_tag = f"_slider_TF{i}"
                if dpg.does_item_exist(slider_tag):
                    dpg.set_value(slider_tag, new_val)
                with torch.no_grad():
                    self.render_kwargs["dict_params"]["opacity_factors"][i].opacity_factor = torch.tensor(
                        new_val, dtype=torch.float32, device="cuda"
                    )
            self.need_update = True
            self.step()  # Update the GUI with the new opacities
            print("Updated TF opacities for target TFs:", tf_numbers)

            # ===== Step 3: Grab the masked rendered image =====
            # At this point self.save_rgba_buffer is assumed to have been updated (via your GUI) to reflect these opacity changes.
            rendered_img = self.save_rgba_buffer  # Expected shape: (H, W, 3) or (H, W, 4)
            rendered_uint8 = (rendered_img * 255).astype(np.uint8)
            if rendered_uint8.shape[-1] == 4:
                masked_orig = Image.fromarray(rendered_uint8, mode="RGBA")
            else:
                masked_orig = Image.fromarray(rendered_uint8, mode="RGB").convert("RGBA")
            print("Converted masked rendered image to a PIL image in memory.")

            # Extract the alpha channel (which acts as a mask for the target TFs)
            mask_alpha = masked_orig.split()[3]
            rgb_masked = masked_orig.convert("RGB")

            # Convert the masked image to a tensor for ip2p processing.
            img_tensor = transforms.ToTensor()(rgb_masked).unsqueeze(0)  # Shape: [1, 3, H, W]
            img_tensor = img_tensor.to(torch.float16)

            # ===== Step 4: Run the ip2p edit on the masked image =====
            text_emb = self.ip2p.pipe._encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=""
            )

            with torch.no_grad():
                edited = self.ip2p.edit_image(
                    text_embeddings=text_emb,
                    image=img_tensor.to(device),
                    image_cond=img_tensor.to(device),
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    diffusion_steps=diffusion_steps,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                )

            stylized = edited.squeeze(0).clamp(0, 1).cpu()  # [3, H, W]
            stylized_rgb = transforms.ToPILImage()(stylized)
            stylized_rgba = stylized_rgb.convert("RGBA")
            # Resize the mask alpha to match the stylized image and attach it.
            mask_alpha = mask_alpha.resize(stylized_rgba.size)
            stylized_rgba.putalpha(mask_alpha)
            stylized_rgba.save("stylized_components.png")
            print("Generated stylized semantic components in memory based on target TFs.")

            # ===== Step 5: Restore original opacities =====
            for i, orig_val in ori_tf_opacity.items():
                slider_tag = f"_slider_TF{i}"
                if dpg.does_item_exist(slider_tag):
                    dpg.set_value(slider_tag, orig_val)
                with torch.no_grad():
                    self.render_kwargs["dict_params"]["opacity_factors"][i].opacity_factor = torch.tensor(
                        orig_val, dtype=torch.float32, device="cuda"
                    )
            self.need_update = True
            self.step()  # Update the GUI with the restored opacities
            print("Restored original TF opacities.")

            # ===== Step 6: Composite the stylized result onto the full rendered image =====
            # Get the full (restored) rendered image.
            full_rendered_img = self.save_rgba_buffer
            full_rendered_uint8 = (full_rendered_img * 255).astype(np.uint8)
            if full_rendered_uint8.shape[-1] == 4:
                full_orig = Image.fromarray(full_rendered_uint8, mode="RGBA")
            else:
                full_orig = Image.fromarray(full_rendered_uint8, mode="RGB").convert("RGBA")
            full_orig = full_orig.convert("RGBA")

            # Composite the stylized (target) image over the full original image using the alpha channel.
            composite_img = Image.alpha_composite(full_orig, stylized_rgba)
            composite_img.save("stylized_scene.png")
            print("Saved composite stylization image.")

            # ===== Step 7: Update the GUI texture =====
            converted_rgb = composite_img.convert("RGB")
            new_img = np.array(converted_rgb, dtype=np.float32) / 255.0
            new_img = np.ascontiguousarray(new_img)  # Ensure contiguous memory
            self.render_buffer = self.overlay_legend(new_img, self.legend_dict)
            dpg.set_value("_texture", self.render_buffer)
            self.need_update = False
            print("GUI updated with the composite stylized image.")

            self.append_chat_bubble("System", "Stylization process has completed. The updated image is now displayed.")

        else:
            # -------- Original processing when tf_numbers is None --------
            rendered_img = self.save_rgba_buffer  # Expected shape: (H, W, 3) or (H, W, 4)
            rendered_uint8 = (rendered_img * 255).astype(np.uint8)
            if rendered_uint8.shape[-1] == 4:
                orig = Image.fromarray(rendered_uint8, mode="RGBA")
            else:
                orig = Image.fromarray(rendered_uint8, mode="RGB").convert("RGBA")
            print("Converted rendered image to a PIL image in memory.")

            alpha_channel = orig.split()[3]
            rgb_orig = orig.convert("RGB")

            img_tensor = transforms.ToTensor()(rgb_orig).unsqueeze(0)  # Shape: [1, 3, H, W]
            img_tensor = img_tensor.to(torch.float16)

            text_emb = self.ip2p.pipe._encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=""
            )

            with torch.no_grad():
                edited = self.ip2p.edit_image(
                    text_embeddings=text_emb,
                    image=img_tensor.to(device),
                    image_cond=img_tensor.to(device),
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    diffusion_steps=diffusion_steps,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                )

            stylized = edited.squeeze(0).clamp(0, 1).cpu()  # [3, H, W]
            stylized_rgb = transforms.ToPILImage()(stylized)
            stylized_rgba = stylized_rgb.convert("RGBA")
            alpha_channel = alpha_channel.resize(stylized_rgba.size)
            stylized_rgba.putalpha(alpha_channel)
            stylized_rgba.save("stylized_scene.png")
            print("Generated stylized image in memory.")

            converted_rgb = stylized_rgba.convert("RGB")
            new_img = np.array(converted_rgb, dtype=np.float32) / 255.0
            new_img = np.ascontiguousarray(new_img)

            self.render_buffer = self.overlay_legend(new_img, self.legend_dict)

            dpg.set_value("_texture", self.render_buffer)
            self.need_update = False
            print("GUI updated with the stylized image.")

            self.append_chat_bubble("System", "Stylization process has completed. The updated image is now displayed.")


    def on_llm_selected(self, sender, app_data):
        """Handles LLM selection change."""
        new_model = app_data  # The selected model
        self.llm_name = new_model  # Update internal model name
        dpg.set_value("llm_selector", new_model)  # Update UI

        # Switch API key settings if using DeepSeek
        if 'deepseek' in self.llm_name.lower():
            self.llm_client = OpenAI(api_key=self.args.api_key[self.llm_name], base_url="https://api.deepseek.com")
        elif 'gpt' in self.llm_name.lower():
            self.llm_client = OpenAI(api_key=self.args.api_key[self.llm_name])
        else:
            self.llm_client = OpenAI(api_key=self.args.api_key[self.llm_name], base_url="https://api.llama-api.com")

        # Log the change
        self.append_chat_bubble("System", f"Switched LLM to {new_model}")


    def on_chat_send_clicked(self, sender, app_data):
        # Get user text
        user_text = dpg.get_value("chat_input").strip()
        if not user_text:
            return
        
        # Add to chat log (user side)
        self.append_chat_bubble("User", user_text)
        dpg.set_value("chat_input", "")  # clear input
        dpg.add_text(f"Step {self.query_step}:\n", parent='command_history')
        dpg.bind_item_font(dpg.last_item(), self.chat_font)
        dpg.set_y_scroll('command_history', -1.0)
        dpg.add_text(f"Step {self.query_step}:\n", parent='query_history')
        dpg.bind_item_font(dpg.last_item(), self.chat_font)
        dpg.set_y_scroll('query_history', -1.0)

        # Run LLM processing in a separate thread
        threading.Thread(target=self.process_llm_query, args=(user_text,), daemon=True).start()
        self.append_chat_bubble("System", "Processing...")

    def record_audio(self):
        """Continuously record audio until stopped."""
        print("Recording started...")
        self.audio_data = []
        with sd.InputStream(samplerate=self.fs, channels=1, callback=self.audio_callback):
            while self.is_recording:
                sd.sleep(100)  # Small sleep to prevent thread blocking
        print("Recording stopped.")

    def audio_callback(self, indata, frames, time, status):
        """Callback function to collect audio chunks."""
        if status:
            print("Recording Error:", status)
        self.audio_data.append(indata.copy())

    def save_and_transcribe_audio(self):
        """Save recorded audio to a file and transcribe it using OpenAI Whisper."""
        audio_filename = "user_audio.wav"
        audio_array = np.concatenate(self.audio_data, axis=0)
        sf.write(audio_filename, audio_array, self.fs)

        print("Audio recorded, transcribing...")
        with open(audio_filename, "rb") as audio_file:
            transcript = self.audio_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
        print("Transcription:", transcript)
        return transcript.text if isinstance(transcript, dict) and "text" in transcript else transcript

    def on_audio_record_clicked(self, sender, app_data):
        """Toggle audio recording on button click."""
        pygame.mixer.music.stop()
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            dpg.configure_item("audio_record_button", label="Listening...")
            self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.recording_thread.start()
        else:
            # Stop recording and transcribe
            self.is_recording = False
            dpg.configure_item("audio_record_button", label="Tap to Speak")
            if self.recording_thread:
                self.recording_thread.join()  # Ensure recording thread stops
            transcript = self.save_and_transcribe_audio()
            if transcript:
                self.append_chat_bubble("User", transcript)
                dpg.set_value("chat_input", "")  # clear text input
                dpg.add_text(f"Step {self.query_step}:\n", parent='command_history')
                dpg.bind_item_font(dpg.last_item(), self.chat_font)
                dpg.set_y_scroll('command_history', -1.0)
                dpg.add_text(f"Step {self.query_step}:\n", parent='query_history')
                dpg.bind_item_font(dpg.last_item(), self.chat_font)
                dpg.set_y_scroll('query_history', -1.0)
                threading.Thread(target=self.process_llm_query, args=(transcript,), daemon=True).start()

    def text_to_speech(self, text):
        """Convert text to speech and play the resulting audio."""
        if self.audio_mute:
            # When mute, skip TTS output.
            print("Audio is mute; skipping TTS playback.")
            return
        audio_file = "assistant_response.mp3"
        response = self.audio_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            speed=1.2
        )
        # Manually write the audio stream to a file
        with open(audio_file, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

        # Play audio in a separate thread to avoid blocking the GUI
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
        except Exception as e:
            print("Error playing audio:", e)


    def process_llm_query(self, user_text):
        """Runs LLM query in a separate thread with an iterative refinement loop."""
        max_refinements = 15 # maximum number of refinement iterations
        iteration = 0
        buffer = BytesIO()
        img = Image.fromarray((self.render_buffer * 255).astype('uint8'))
        img.save(buffer, format="PNG")
        current_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        while iteration < max_refinements:
            self.append_command_log(f"------------Iteration {iteration}------------")
            part1_commands, part2_explanations, iterate_flag, best_tf, debug_text, open_vocab_text, self.conversation_history_parser, self.conversation_history_controller = process_user_query(
                user_text,
                self.conversation_history_parser,
                self.conversation_history_controller,
                self.query_step,
                iteration,
                self.llm_client,
                self.clip_model,
                self.tf_embeddings,
                self.llm_name,
                self.img_path,
                self.dataset_info,
                current_image_base64
            )
            #print(debug_text)
            
            if open_vocab_text != "":
                for line in open_vocab_text.split("\n"):
                    dpg.add_text(line, parent='query_history')
                    dpg.bind_item_font(dpg.last_item(), self.chat_font)
                    dpg.set_y_scroll('query_history', -1.0)
            else:
                dpg.add_text("None\n", parent='query_history')
                dpg.bind_item_font(dpg.last_item(), self.chat_font)
                dpg.set_y_scroll('query_history', -1.0)

            for cmd in part1_commands:
                self.process_message(cmd, None)

            for line in part2_explanations:
                self.append_chat_bubble("Assistant", line)

            if iterate_flag == "NO":
                full_response = " ".join(part2_explanations)
                self.text_to_speech(full_response)
                break
            else:
                iteration += 1
                # Optionally, update conversation history to request refinement
                self.conversation_history_controller.append({
                    "role": "user",
                    "content": f"Please refine your previous instructions based on the updated visualization. (Iteration {iteration})"
                })
                # Capture the updated visualization image after the commands have been executed
                buffer = BytesIO()
                img = Image.fromarray((self.save_rgba_buffer * 255).astype('uint8'))
                img.save(buffer, format="PNG")
                current_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                full_response = " ".join(part2_explanations)
                self.text_to_speech(full_response)
            self.append_command_log('')
        self.query_step += 1

    def append_chat_bubble(self, speaker: str, message: str):
        """Create a bubble with bigger font and different themes for user/assistant/system."""
        def wrap_text(text, max_chars_per_line):
            """
            Wraps text manually based on the number of characters per line.
            If a newline ("\n") is encountered in the input, it forces a new line.
            """
            lines = []
            # Split the text by newline to handle existing line breaks
            paragraphs = text.split("\n")
            for paragraph in paragraphs:
                # For each paragraph, wrap the words.
                words = paragraph.split()
                current_line = ""
                for word in words:
                    # If current_line is empty, start with the word.
                    if not current_line:
                        current_line = word
                    # Else, if adding the word fits within the max length, append it.
                    elif len(current_line) + 1 + len(word) <= max_chars_per_line:
                        current_line += " " + word
                    else:
                        # Otherwise, append the current line and start a new one.
                        lines.append(current_line)
                        current_line = word
                # If there's any remaining text in current_line, append it.
                if current_line:
                    lines.append(current_line)
            return lines
        
        with dpg.group(parent="chat_history", horizontal=True) as row_id:
            
            # If user, push bubble to right; else minimal left gap
            if speaker.lower() == "user":
                dpg.add_spacer(width=int(self.chatW * 0.18))
            else:
                dpg.add_spacer(width=1)
            
            lines = wrap_text(f"{speaker}: {message}", max_chars_per_line=36)
            estimated_height = 22 * len(lines) + 6

            # A fixed bubble size for demonstration:
            with dpg.child_window(
                menubar=False,
                border=False,
                autosize_x=False,
                autosize_y=False,
                width=int(self.chatW * 0.7)+20,
                height=estimated_height
            ) as bubble_id:
                # Add text with line-wrap
                text_tag = dpg.add_text(
                    "\n".join(lines)
                )
                # Make the text bigger by binding the bubble_font
                dpg.bind_item_font(text_tag, self.bubble_font)

            # Choose theme by speaker
            if speaker.lower() == "user":
                dpg.bind_item_theme(bubble_id, self.user_bubble_theme)
            elif speaker.lower() == "assistant":
                dpg.bind_item_theme(bubble_id, self.assistant_bubble_theme)
            else:
                # e.g. "system" or anything else
                dpg.bind_item_theme(bubble_id, self.system_bubble_theme)

        # Auto-scroll to the bottom
        dpg.set_y_scroll("chat_history", -1.0)

    def append_command_log(self, command, line_length=25):
        """Logs commands sent by the agent to the GUI."""
        def wrap_text(text, line_length):
            words = text.split()
            wrapped_lines = []
            line = ""
            for word in words:
                if len(line) + len(word) + 1 <= line_length:
                    line += (word + " ")
                else:
                    wrapped_lines.append(line.strip())
                    line = word + " "
            if line:
                wrapped_lines.append(line.strip())
            return "\n".join(wrapped_lines)

        wrapped_command = wrap_text(command, line_length)
        if command.startswith("--") or command == "":
            new_line = f"{command}\n"
        else:
            new_line = f"> {wrapped_command}\n"

        dpg.add_text(new_line, parent='command_history')
        dpg.bind_item_font(dpg.last_item(), self.chat_font)
        dpg.set_y_scroll('command_history', -1.0)

    def register_dpg(self):

        with dpg.theme() as self.white_bg_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (240, 240, 240, 255))  # RGBA for white
        
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.imgW, self.imgH, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.imgW, height=self.imgH, pos=(self.ctrlW, 0), no_resize=True,
                        no_move=True, no_title_bar=True, no_background=False):
            # add the texture
            dpg.add_image("_texture")
        dpg.bind_item_theme("_primary_window", self.white_bg_theme)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=self.ctrlW, height=self.imgH, pos=(0, 0),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=False):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Inference Time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True, leaf=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = self.inv_menu_map[app_data]
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Mode")
                    dpg.add_combo(self.menu, indent=self.widget_top, label='', default_value="Blinn-Phong", callback=callback_change_mode, tag="_combo_mode")

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Field of View")
                    dpg.add_slider_int(label="",tag="_slider_fovy",indent=self.widget_top, min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)
                    
                def callback_set_BG_color(sender, app_data):
                    bg_color = app_data[:3]
                    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    self.render_kwargs["bg_color"] = bg_color
                    self.need_update = True
                  
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Background Color")
                    dpg.add_color_edit(label="", tag="_color_edit_background", no_alpha=True, default_value=[255, 255, 255],
                                       indent=self.widget_top, callback=callback_set_BG_color) 
                
                def callback_reset_view(sender, app_data):
                    file_path = os.path.join(self.img_path, "initial_view.txt")
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "r") as f:
                                lines = f.readlines()
                            # Find index for the "Radius:" line.
                            radius_index = None
                            for i, line in enumerate(lines):
                                if line.startswith("Radius:"):
                                    radius_index = i
                                    break
                            if radius_index is None or radius_index < 2:
                                print("Initial view file format is incorrect.")
                            else:
                                # Join the lines that contain the rotation matrix (from index 1 to radius_index).
                                rot_str = " ".join(line.strip() for line in lines[1:radius_index])
                                rot_matrix = np.array(ast.literal_eval(rot_str))
                                rot_obj = R.from_matrix(rot_matrix)
                                
                                # Parse the radius.
                                radius_line = lines[radius_index].strip()  # e.g., "Radius: 3.305785123966942"
                                radius = float(radius_line.split("Radius:")[1].strip())
                                
                                # Parse the center from the next line.
                                center_line = lines[radius_index+1].strip()  # e.g., "Center: [ 0.06522223, -0.02958885,  0.00098899]"
                                center = np.array(ast.literal_eval(center_line.split("Center:")[1].strip()))
                                
                                # Set the camera parameters.
                                self.cam.rot = rot_obj
                                self.cam.radius = radius
                                self.cam.center = center
                                
                                print("Loaded initial view from", file_path)
                        except Exception as e:
                            print("Error loading initial view:", e)
                            self.cam.reset_view()
                    else:
                        print("Initial view file not found:", file_path)
                        self.cam.reset_view()
                    if self.freeze_view:
                        dpg.configure_item("_freeze_view_button", label="Freeze View")
                        print("View unfrozen.")
                        self.freeze_view = False
                    self.need_update = True
                
                def callback_save_image(sender, app_data):
                    rendered_img = self.save_rgba_buffer
                    rendered_img = (rendered_img*255).astype(np.uint8)[...,[2,1,0,3]]
                    # cv2.imwrite(os.path.join("./GUI_results", f'rendered_img.png'), rendered_img)
                    # Get current timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save the image with the timestamp in the filename
                    filename = f'./screenshots/user0/rendered_img_{timestamp}.png'
                    cv2.imwrite(filename, rendered_img)
                    print("Image Saved")
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset View", tag="_button_reset_view", width=self.ctrlW//3-10, callback=callback_reset_view)
                                    # Add a button that works as an icon with label "Freeze View"
                    dpg.add_button(label="Freeze View",tag="_freeze_view_button",width=self.ctrlW//3-10,callback=self.on_freeze_view)
                    dpg.add_button(label="Save Image", tag="_button_save_image",width=self.ctrlW//3-10, callback=callback_save_image)
                    dpg.bind_item_theme("_button_reset_view", self.theme_button)
                    dpg.bind_item_theme("_button_save_image", self.theme_button)
                    dpg.bind_item_theme("_freeze_view_button", self.theme_button)
                                    
            # color & opacity editing
            with dpg.collapsing_header(label="Color & Opacity Editing", default_open=True, leaf=True):
                    with dpg.group(horizontal=True, horizontal_spacing=0):
                        for i in range(self.TFnums):
                            self.add_oneTFSlider(i)
                    def callback_reset_color_opacity(sender, app_data):
                        with torch.no_grad():
                            for TFidx in range(self.TFnums):
                                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                                self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                                color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                                dpg.set_value(f"_slider_TF{TFidx}", 1)
                                dpg.set_value(f"_color_TF{TFidx}", tuple(color_value))
                        self.legend_dict = {}
                        self.need_update = True

                    def callback_reset_all(sender, app_data):
                        # reset fov
                        dpg.set_value("_slider_fovy", self.default_fov)
                        self.cam.fovy = self.default_fov
                        # reset color and opacity
                        with torch.no_grad():
                            for TFidx in range(self.TFnums):
                                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                                self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                                color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                                dpg.set_value(f"_slider_TF{TFidx}", 1)
                                dpg.set_value(f"_color_TF{TFidx}", tuple(color_value))
                        
                        # reset view
                        file_path = os.path.join(self.img_path, "initial_view.txt")
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, "r") as f:
                                    lines = f.readlines()
                                # Find index for the "Radius:" line.
                                radius_index = None
                                for i, line in enumerate(lines):
                                    if line.startswith("Radius:"):
                                        radius_index = i
                                        break
                                if radius_index is None or radius_index < 2:
                                    print("Initial view file format is incorrect.")
                                else:
                                    # Join the lines that contain the rotation matrix (from index 1 to radius_index).
                                    rot_str = " ".join(line.strip() for line in lines[1:radius_index])
                                    rot_matrix = np.array(ast.literal_eval(rot_str))
                                    rot_obj = R.from_matrix(rot_matrix)
                                    
                                    # Parse the radius.
                                    radius_line = lines[radius_index].strip()  # e.g., "Radius: 3.305785123966942"
                                    radius = float(radius_line.split("Radius:")[1].strip())
                                    
                                    # Parse the center from the next line.
                                    center_line = lines[radius_index+1].strip()  # e.g., "Center: [ 0.06522223, -0.02958885,  0.00098899]"
                                    center = np.array(ast.literal_eval(center_line.split("Center:")[1].strip()))
                                    
                                    # Set the camera parameters.
                                    self.cam.rot = rot_obj
                                    self.cam.radius = radius
                                    self.cam.center = center
                                    
                                    print("Loaded initial view from", file_path)
                            except Exception as e:
                                print("Error loading initial view:", e)
                                self.cam.reset_view()
                        else:
                            print("Initial view file not found:", file_path)
                            self.cam.reset_view()
                        if self.freeze_view:
                            dpg.configure_item("_freeze_view_button", label="Freeze View")
                            print("View unfrozen.")
                            self.freeze_view = False

                        # remove legends
                        self.legend_dict = {}

                        self.need_update = True

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset Color & Opacity", tag="_button_reset_color_opacity",width=(self.ctrlW-15)/2, callback=callback_reset_color_opacity)
                        dpg.add_button(label="Reset All", tag="_button_reset_all",width=(self.ctrlW-15)/2, callback=callback_reset_all)
                        dpg.bind_item_theme("_button_reset_color_opacity", self.theme_button)
                        dpg.bind_item_theme("_button_reset_all", self.theme_button)
                        # dpg.bind_item_theme("_button_save_color_opacity", self.theme_button)
                            
            # light editing
            with dpg.collapsing_header(label="Light Editing", default_open=True, leaf=True):
                #* Use Headlight button
                def callback_headlight(sender, app_data):
                    if app_data == False:
                        self.useHeadlight = app_data
                        self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    else:
                        self.useHeadlight = app_data
                    self.render_kwargs["dict_params"]["light_transform"].useHeadLight = self.useHeadlight
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Headlight")
                    dpg.add_checkbox(label="", tag="_checkbox_headlight", callback=callback_headlight, default_value=self.useHeadlight)
                
                #* Light angle and elevation sliders
                def callback_light_angle(sender, app_data):
                    if self.useHeadlight:
                        return
                    if sender == "_slider_light_angle":
                        self.light_angle = app_data
                    else:
                        self.light_elevation = app_data
                    
                    self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Azimuthal")
                    dpg.add_slider_int(label="", tag="_slider_light_angle", indent=self.widget_indent,
                                       default_value=self.light_angle, min_value=-180, max_value=180, callback=callback_light_angle)
                with dpg.group(horizontal=True):
                    dpg.add_text("Polar")
                    dpg.add_slider_int(label="", tag="_slider_light_elevation", indent=self.widget_indent,
                                       default_value=self.light_elevation, min_value=-90, max_value=90, callback=callback_light_angle)
                
                #* ambient sldiers
                def callback_light_multi(sender, app_data):
                    if sender == "_slider_ambient_multi":
                        self.render_kwargs["dict_params"]["light_transform"].ambient_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_light_intensity_multi":
                        self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_specular_multi":
                        self.render_kwargs["dict_params"]["light_transform"].specular_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_shininess_multi":
                        self.render_kwargs["dict_params"]["light_transform"].shininess_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    self.need_update = True
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Ambient")
                    dpg.add_slider_float(label="", tag="_slider_ambient_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Diffuse")
                    dpg.add_slider_float(label="", tag="_slider_light_intensity_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Specular")
                    dpg.add_slider_float(label="", tag="_slider_specular_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Shininess")
                    dpg.add_slider_float(label="", tag="_slider_shininess_multi", indent=self.widget_indent, default_value=3, min_value=1, max_value=5, callback=callback_light_multi)
            
            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        # Bind the theme to the window
        dpg.bind_item_theme("_control_window", self.white_bg_theme)

        ### register camera handler
        def callback_camera_start_rotate(sender, app_data):
            self.rotating = True

        def callback_camera_drag_rotate(sender, app_data):
            if self.freeze_view:
                return

            if not dpg.is_item_focused("_primary_window"):
                return
            MouseX, MouseY = dpg.get_mouse_pos()
            x = -(MouseX/ self.imgW - 0.5) * 2
            y = -(MouseY/ self.imgH - 0.5) * 2

            # self.cam.orbit(dx, dy)
            if(self.prev_mouseX is None or self.prev_mouseY is None):
                self.prev_mouseX = x
                self.prev_mouseY = y
                return
            if (self.rotating):
                self.cam.orbit(self.prev_mouseX, self.prev_mouseY, x, y)
                self.prev_mouseX = x
                self.prev_mouseY = y
            
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                
        
        def callback_camera_end_rotate(sender, app_data):
            self.rotating = False
            self.prev_mouseX = None
            self.prev_mouseY = None
        
        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):
            if self.freeze_view:
                return

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
        #* KT: modifed this to use ArcBall Rotation
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_start_rotate)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)

            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_end_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='NLI4VolVis', width=self.imgW+self.ctrlW+self.chatW+self.commandW, height=self.imgH, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-vo', '--view_config', default=None, required=False, help="the config root")
    parser.add_argument('--image_path', type=str, default=None, required=True, help="the original training image dir")
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument('-t', '--type', choices=['inverse','phong'], default='inverse')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="resume from checkpoint")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--gui_debug", action="store_true", help="show debug info in GUI")
    parser.add_argument("--stylize_name",type=str, default=None, help="the edit name of the stylized model you want to load")
    parser.add_argument("--api_key", type=str, required=True, 
                        help="JSON string containing API keys for multiple LLMs")
    parser.add_argument("--llm_name", type=str, default="gpt-4o",
                        help="Name of the LLM model to use (e.g. gpt-3.5-turbo, gpt-4, gpt-4o)")
    parser.add_argument("--embedding_name", type=str, default="image_filtered_embedding_entropy.npy",
                        help="Name of the embedding .npy file in each TF directory.")

    args = parser.parse_args()
    # Convert API key JSON string to dictionary
    args.api_key = json.loads(args.api_key)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    
    pbr_kwargs = dict()
    scene_dict = load_ckpts_paths(args.source_dir, args.stylize_name)
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)
    palette_color_transforms = []
    opacity_transforms = []
    TFcount=0
    for TFs_name in TFs_names:
        
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.create_from_ckpt(f"{scene_dict[TFs_name]['palette']}")
        palette_color_transforms.append(palette_color_transform)
        # ic(TFcount)
        opacity_factor=0.0 if TFcount not in [] else 1.0
        opacity_transform = LearningOpacityTransform(opacity_factor=opacity_factor)
        opacity_transforms.append(opacity_transform)
        TFcount+=1
        
        
    light_transform = LearningLightTransform(theta=180, phi=0)
    # load gaussians
    gaussians_composite = scene_composition(scene_dict, dataset)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians_composite,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "dict_params": {
            "palette_colors": palette_color_transforms,
            "opacity_factors": opacity_transforms,
            "light_transform": light_transform
        }
    }
    
    # ic(scene_dict)
    # ic(checkpoints)
        
    render_fn = render_fn_dict[args.type]
    
    #* remove this if remove --vo argument
    H, W = 800, 800
    fovx = 30 * np.pi / 180
    fovy = focal2fov(fov2focal(fovx, W), H)
    # fovy = 30.5 * np.pi / 180
    if args.view_config is None:
        c2w = np.array([
            [0.0, 0.0, -1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    else:
        view_config_file = f"{args.view_config}/transforms_test.json"
        view_dict = load_json_config(view_config_file)
        all_views = view_dict["frames"]
        #todo: add view index
        c2w = np.array(all_views[0]["transform_matrix"]).reshape(4, 4) 
        c2w /= 2
        c2w[:3, 1:3] *= -1
    
    windows = GUI(H, W, fovy,
                  c2w=c2w, center=np.zeros(3),
                  render_fn=render_fn, render_kwargs=render_kwargs, TFnums=TFs_nums, args=args,
                  mode=args.type, debug=args.gui_debug)
    
    while dpg.is_dearpygui_running():
        windows.render()