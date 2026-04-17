import numpy as np
import cv2
import platform

import uvicorn
uvicorn.config.LOGGING_CONFIG = None
import gradio.http_server as http_server
import uvicorn.config as uvicorn_config
_original_config_init = uvicorn_config.Config.__init__

def _patched_config_init(self, app, *args, **kwargs):
    if 'log_config' not in kwargs:
        kwargs['log_config'] = None
    _original_config_init(self, app, *args, **kwargs)

uvicorn_config.Config.__init__ = _patched_config_init

import gradio as gr
from rknnlite.api import RKNNLite

# decice tree for RK356x/RK3576/RK3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'RK3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

# RKNN 模型路径
# lubancat-0/1/2 系列 对应 RK3566/RK3568
# lubancat-3/4/5 系列 分别对应 RK3576/RK3588
RK3588_RKNN_MODEL = './model/yolo26n_for_rk3588.rknn'
RK3576_RKNN_MODEL = './model/yolo26n_for_rk3576.rknn'
RK3566_RK3568_RKNN_MODEL = './model/yolo26n_for_rk3566_rk3568.rknn'
RK3562_RKNN_MODEL = './model/yolo26n_for_rk3562.rknn'

OBJ_THRESH = 0.25
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

rknn_lite = None

def init_rknn():
    """初始化 RKNN 模型"""
    global rknn_lite

    if rknn_lite is not None:
        return True, "RKNN model already loaded"

    # Get device information
    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3576':
        rknn_model = RK3576_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        return False, "This demo cannot run on the current platform: {}".format(host_name)

    rknn_lite = RKNNLite()

    # Load RKNN model
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        return False, 'Load RKNN model failed'

    # Init runtime environment
    if host_name in ['RK3576', 'RK3588']:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        return False, 'Init runtime environment failed'

    return True, "RKNN model loaded successfully"

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """
    等比例缩放图像并添加灰边填充

    参数:
        im: 输入图像 (H, W, C)
        new_shape: 目标尺寸 (H, W)
        color: 填充颜色 (B, G, R)

    返回:
        im: letterbox 后的图像
        ratio: 缩放比例
        (dw, dh): 填充的宽高
    """
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算填充
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def get_real_box(src_shape, boxes, dw, dh, ratio):
    """
    将检测框从 letterbox 后的坐标还原到原图坐标

    参数:
        src_shape: 原图形状 (H, W)
        boxes: (N, 4) 检测框 [x1, y1, x2, y2]
        dw, dh: letterbox 填充的宽高
        ratio: letterbox 缩放比例

    返回:
        boxes: 还原后的检测框
    """
    # x 坐标 (索引 0, 2) 减去 dw, y 坐标 (索引 1, 3) 减去 dh
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    # 裁剪到原图范围内
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_shape[0])

    return boxes

def postprocess_yolo26(outputs):
    """
    后处理 - 三尺度输出解码

    参数:
        outputs: (1, 84, 80, 80), (1, 84, 40, 40), (1, 84, 20, 20)

    返回:
        boxes: (N, 4) - [x1, y1, x2, y2] 归一化到原图
        scores: (N,) - 置信度
        classes: (N,) - 类别索引
    """

    all_boxes, all_scores, all_classes = [], [], []

    # strides for 3 scales
    strides = [8, 16, 32]

    for i, output in enumerate(outputs):
        # output shape: (1, 84, h, w) -> (84, h*w)
        pred = output[0].reshape(84, -1)

        h, w = output.shape[2], output.shape[3]
        stride = strides[i]

        # anchor_points
        y = np.arange(h) * stride + stride // 2
        x = np.arange(w) * stride + stride // 2
        xx, yy = np.meshgrid(x, y)
        anchor_points = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, N)

        #box cls_scores
        box_dist = pred[:4, :]  # (4, N)
        cls_scores = pred[4:, :]  # (80, N)

        # dist2bbox
        x1y1 = anchor_points - box_dist[:2, :] * stride
        x2y2 = anchor_points + box_dist[2:, :] * stride
        boxes = np.concatenate([x1y1, x2y2], axis=0)  # (4, N)

        # max_cls_scores
        max_cls_scores = cls_scores.max(axis=0)  # (N,)

        mask = max_cls_scores > OBJ_THRESH
        if not mask.any():
            continue

        # classes
        classes = cls_scores.argmax(axis=0)

        all_boxes.append(boxes[:, mask])
        all_scores.append(max_cls_scores[mask])
        all_classes.append(classes[mask])

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    boxes = np.concatenate(all_boxes, axis=1).T  # (N, 4)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    return boxes, scores, classes

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[int(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    return image

def detect_image(input_image, confidence_threshold):
    """对输入图片进行目标检测"""
    global rknn_lite

    # 初始化模型
    success, message = init_rknn()
    if not success:
        return None, message

    if input_image is None:
        return None, "请先上传一张图片"

    # BGR for cv2
    img_src = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    src_shape = img_src.shape[:2]

    # preprocess
    img, ratio, (dw, dh) = letterbox(img_src, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # inference
    outputs = rknn_lite.inference(inputs=[img])

    # postprocess
    input_data = [outputs[0], outputs[1], outputs[2]]
    boxes, scores, classes = postprocess_yolo26(input_data)

    if len(boxes) == 0:
        return input_image, "未检测到任何目标"

    # 根据置信度过滤
    mask = scores > confidence_threshold
    if not mask.any():
        return input_image, f"没有置信度大于 {confidence_threshold} 的目标"

    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    # 转换回原图坐标
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_shape[0])
    
    # 绘制结果
    result = draw(img_src, boxes, scores, classes)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # 生成检测信息
    info = "检测到的目标:\n"
    info += "-" * 40 + "\n"
    for box, score, cl in zip(boxes, scores, classes):
        info += f"{CLASSES[int(cl)]}: {score:.2f}  "
        info += f" [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]\n"

    return result, info

def create_demo():

    with gr.Blocks(title="YOLO26 目标检测") as demo:
        gr.Markdown("""
        # YOLO26 目标检测

        基于 RKNN 的 YOLO26 目标检测模型，上传图片即可进行目标检测。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入")
                input_image = gr.Image(label="上传图片", type="numpy", sources=["upload"])
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="置信度阈值"
                )
                detect_btn = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 输出")
                output_image = gr.Image(label="检测结果")
                info_text = gr.Textbox(label="检测信息", lines=10)

        # 示例图片
        gr.Markdown("### 示例图片")
        gr.Examples(
            examples=[
                ["./model/bus.jpg"],
            ],
            inputs=[input_image],
        )

        detect_btn.click(
            fn=detect_image,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, info_text]
        )

        input_image.change(
            fn=detect_image,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, info_text]
        )

    return demo

if __name__ == '__main__':
    demo = create_demo()
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        allowed_paths=['./model']
    )
