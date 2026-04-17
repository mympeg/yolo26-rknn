# YOLO26

基于RKNN的YOLO26目标检测模型，使用Gradio构建Web演示界面。

## 快速开始

### 1. 安装依赖

```bash
cd yolo26/
source .venv/bin/activate
```

### 2. 准备模型

确保 `model/` 目录下存在对应平台的 RKNN 模型文件：

```
model/
├── yolo26n_for_rk3566_rk3568.rknn
├── yolo26n_for_rk3562.rknn
├── yolo26n_for_rk3576.rknn
├── yolo26n_for_rk3588.rknn
└── bus.jpg  # 示例图片
```

### 3. 运行程序

```bash
python main.py
```

如果安装了uv也可以直接运行命令：

```bash
uv run main.py
```

启动成功后，在浏览器中访问：`http://<设备 IP>:7860`

## 使用说明

1. 上传待检测图片，或点击示例图片加载测试图
2. 调整置信度阈值（默认 0.25）
3. 点击"开始检测"按钮
4. 查看检测结果和检测信息

## 链接

https://doc.embedfire.com/linux/rk356x/Python/zh/latest/ai/yolo26..html
