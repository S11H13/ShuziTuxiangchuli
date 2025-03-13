# 高级求余条纹特效图像处理

此项目包含一个Jupyter Notebook文件`kaiqiny.ipynb`，用于对图像应用高级求余条纹特效。该特效通过调整图像的亮度和饱和度，生成具有条纹效果的图像。

## 文件结构

- `earth.jpg`: 示例输入图像文件。
- `kaiqiny.ipynb`: 包含图像处理代码的Jupyter Notebook文件。

## 依赖项

在运行此Notebook之前，请确保已安装以下Python库：

- `Pillow`: 用于图像处理。
- `numpy`: 用于数值计算。
- `matplotlib`: 用于图像可视化。

可以使用以下命令安装这些库：

```sh
pip install pillow numpy matplotlib
```

## 使用方法

1. 打开`kaiqiny.ipynb`文件。
2. 运行所有代码单元。

## 代码说明

### 导入库

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
```

导入必要的库：`Pillow`用于图像处理，`numpy`用于数值计算，`matplotlib`用于图像可视化。

### 高级求余条纹特效函数

```python
def advanced_mod_effect(img, mod=5, brightness=2.5, saturation=0.3):
    """
    高级求余条纹特效（减弱颜色增强黑白版）
    参数：
    - mod: 条纹周期模数（值越小条纹越密集）
    - brightness: 亮度增益系数
    - saturation: 颜色保留比例（0-1，值越小越接近黑白）
    """
    img_np = np.array(img)
    height, width = img_np.shape[:2]
    
    # 生成动态条纹模板
    x = np.arange(width)
    mask = (x % mod) < (mod // 2)
    
    # 多通道混合增强
    enhanced = np.zeros_like(img_np, dtype=np.float32)
    enhanced[:, mask, 0] = img_np[:, mask, 0] * brightness  # 红色通道增强
    enhanced[:, mask, 1] = img_np[:, mask, 1] * 0.8         # 绿色通道抑制
    enhanced[:, ~mask, 2] = img_np[:, ~mask, 2] * brightness # 蓝色通道交替增强
    
    # 计算灰度并混合颜色通道
    gray = np.dot(enhanced[..., :3], [0.2989, 0.5870, 0.1140])  # RGB转灰度
    gray_3d = gray[..., np.newaxis].repeat(3, axis=2)          # 扩展为三维
    desaturated = enhanced * saturation + gray_3d * (1 - saturation)  # 去饱和混合
    
    # 亮度平衡与裁剪
    luminance = np.clip(desaturated * 1.2, 0, 255).astype(np.uint8)
    
    return luminance
```

该函数实现了高级求余条纹特效。参数说明：

- `mod`: 条纹周期模数，值越小条纹越密集。
- `brightness`: 亮度增益系数。
- `saturation`: 颜色保留比例，值越小越接近黑白。

### 加载并处理图像

```python
# 加载并处理图像
image = Image.open('earth.jpg')
processed_img = advanced_mod_effect(image, mod=4, brightness=2.0, saturation=0.2)
```

加载输入图像`earth.jpg`并应用高级求余条纹特效。

### 可视化处理后的图像

```python
# 大画布可视化
plt.figure(figsize=(10, 10))
plt.imshow(processed_img)
plt.axis('off')
plt.show()
```

使用`matplotlib`库可视化处理后的图像。

## 示例

运行`kaiqiny.ipynb`中的代码后，将显示处理后的图像，具有高级求余条纹特效。
班级：智能科学与技术2班
学号：202352320215

## 许可证

此项目遵循MIT许可证。
