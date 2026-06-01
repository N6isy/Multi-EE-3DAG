# PartNet-Mobility 数据集网页查看器

这个小工具用于在服务器端直接查看 `partnet-mobility-v0.zip` 的数据样式，不需要先完整解压整个 zip。

支持：

- 统计 object 数量、类别分布、常见文件扩展名；
- 浏览 object id / category / 文件数量；
- 查看每个样本的 `meta.json`、`result.json`；
- 解析 `mobility.urdf` 中的 link / joint / axis / limit；
- 浏览样本内部文件树；
- 对 `.obj` mesh 做简单 3D 预览。

> 说明：3D 预览默认通过浏览器从 `unpkg.com` 加载 three.js。如果服务器或浏览器不能联网，JSON、URDF、文件浏览仍然可用，只是 3D 预览会提示无法加载 three.js。

## 1. 放到服务器

把整个 `partnet_mobility_viewer` 文件夹上传到服务器，例如：

```bash
cd /home/lzq/Multi-EE-3DAG/MultiEEAffordance/tools
# 将 partnet_mobility_viewer 放在这里
cd partnet_mobility_viewer
```

## 2. 安装依赖

建议用当前项目环境或单独建一个轻量环境：

```bash
cd /home/lzq/Multi-EE-3DAG/MultiEEAffordance/tools/partnet_mobility_viewer
pip install -r requirements.txt
```

如果服务器不能访问默认 pip 源，可以换清华源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 启动网页

你的 zip 路径是：

```bash
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip
```

直接运行：

```bash
python app.py \
  --zip /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip \
  --host 127.0.0.1 \
  --port 7860
```

或者：

```bash
./run_viewer.sh /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip
```

## 4. 在本地浏览器打开

如果你是在 VSCode Remote SSH / SSH 连接服务器，推荐用 SSH 端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 lzq@你的服务器IP
```

然后在本地浏览器打开：

```text
http://127.0.0.1:7860
```

如果你希望局域网其他机器直接访问，启动时把 host 改成：

```bash
python app.py \
  --zip /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip \
  --host 0.0.0.0 \
  --port 7860
```

然后访问：

```text
http://服务器IP:7860
```

## 5. 常见问题

### 5.1 页面能打开，但 3D 预览失败

这是因为浏览器无法访问 `unpkg.com` 加载 three.js。这个问题不影响 JSON、URDF、文件树查看。

解决方式：

- 临时先只看 JSON/URDF；
- 或者把 three.js 相关文件下载到本地，再把 `static/app.js` 中的 CDN 地址改成本地路径。

### 5.2 object 数量为 0

一般是 zip 内部目录结构不是标准的 `.../<数字object_id>/mobility.urdf` 这类形式。可以先检查：

```bash
python - <<'PY'
import zipfile
p='/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip'
with zipfile.ZipFile(p) as z:
    for name in z.namelist()[:80]:
        print(name)
PY
```

如果 object id 不是数字目录，需要改 `app.py` 中的 `get_object_id_from_zip_path()` 识别逻辑。

### 5.3 zip 很大，启动慢

首次启动会扫描 zip 文件目录并读取部分 `meta.json/result.json` 来统计类别。PartNet-Mobility 规模通常可以接受；如果你只想更快打开，可以把 `app.py` 里 `_load_categories()` 调用临时注释掉。
