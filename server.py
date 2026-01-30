"""
图像几何校正服务器
使用OpenCV + TPS（薄板样条）实现图像Warp功能
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io
import base64
import numpy as np
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.')
CORS(app)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


def grid_warp_image(image, source_points, target_points, gridX, gridY, output_shape):
    """
    基于网格的局部图像warp - 使用三角形仿射变换版本

    将每个网格单元分成两个三角形，分别进行仿射变换
    保证相邻单元之间公共边的连续性
    """
    height, width = output_shape[:2]

    # 创建带alpha通道的结果图像（初始化为原图）
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    print(f"\n=== 网格变形开始 ===")
    print(f"网格: {gridX} x {gridY}, 图像: {width} x {height}")

    move_threshold = 2.0
    warped_triangles = 0

    # 遍历每个网格单元
    for y in range(gridY - 1):
        for x in range(gridX - 1):
            tl = y * gridX + x          # top-left
            tr = y * gridX + (x + 1)    # top-right
            bl = (y + 1) * gridX + x    # bottom-left
            br = (y + 1) * gridX + (x + 1)  # bottom-right

            # 源角点和目标角点
            src_tl = source_points[tl]
            src_tr = source_points[tr]
            src_bl = source_points[bl]
            src_br = source_points[br]

            dst_tl = target_points[tl]
            dst_tr = target_points[tr]
            dst_bl = target_points[bl]
            dst_br = target_points[br]

            # 检查是否有角点移动
            moved = False
            max_move = 0
            for src, dst in [(src_tl, dst_tl), (src_tr, dst_tr), (src_bl, dst_bl), (src_br, dst_br)]:
                dist = np.linalg.norm(src - dst)
                max_move = max(max_move, dist)
                if dist > move_threshold:
                    moved = True

            if not moved:
                continue

            print(f"  变形单元({x},{y}): 最大移动{max_move:.1f}px")

            # 将单元分成两个三角形
            # 三角形1: tl -> tr -> bl (上三角)
            # 三角形2: tr -> br -> bl (下三角)
            triangles = [
                ([src_tl, src_tr, src_bl], [dst_tl, dst_tr, dst_bl]),
                ([src_tr, src_br, src_bl], [dst_tr, dst_br, dst_bl])
            ]

            for src_tri, dst_tri in triangles:
                warped_triangles += 1

                # 计算源三角形边界
                src_tri_arr = np.array(src_tri)
                src_x_min = int(np.floor(np.min(src_tri_arr[:, 0])))
                src_x_max = int(np.ceil(np.max(src_tri_arr[:, 0])))
                src_y_min = int(np.floor(np.min(src_tri_arr[:, 1])))
                src_y_max = int(np.ceil(np.max(src_tri_arr[:, 1])))

                # 边界检查
                src_x_min = max(0, src_x_min)
                src_x_max = min(width, src_x_max)
                src_y_min = max(0, src_y_min)
                src_y_max = min(height, src_y_max)

                if src_x_max <= src_x_min or src_y_max <= src_y_min:
                    continue

                # 计算目标三角形边界
                dst_tri_arr = np.array(dst_tri)
                dst_x_min = int(np.floor(np.min(dst_tri_arr[:, 0])))
                dst_x_max = int(np.ceil(np.max(dst_tri_arr[:, 0])))
                dst_y_min = int(np.floor(np.min(dst_tri_arr[:, 1])))
                dst_y_max = int(np.ceil(np.max(dst_tri_arr[:, 1])))

                # 边界检查
                dst_x_min = max(0, dst_x_min)
                dst_x_max = min(width, dst_x_max)
                dst_y_min = max(0, dst_y_min)
                dst_y_max = min(height, dst_y_max)

                if dst_x_max <= dst_x_min or dst_y_max <= dst_y_min:
                    continue

                # 提取源区域
                src_region = image[src_y_min:src_y_max, src_x_min:src_x_max].copy()

                # 调整源三角形坐标（相对于源区域）
                src_tri_adj = np.array([
                    [src_tri[0][0] - src_x_min, src_tri[0][1] - src_y_min],
                    [src_tri[1][0] - src_x_min, src_tri[1][1] - src_y_min],
                    [src_tri[2][0] - src_x_min, src_tri[2][1] - src_y_min]
                ], dtype=np.float32)

                # 调整目标三角形坐标（相对于目标区域）
                dst_tri_adj = np.array([
                    [dst_tri[0][0] - dst_x_min, dst_tri[0][1] - dst_y_min],
                    [dst_tri[1][0] - dst_x_min, dst_tri[1][1] - dst_y_min],
                    [dst_tri[2][0] - dst_x_min, dst_tri[2][1] - dst_y_min]
                ], dtype=np.float32)

                # 计算仿射变换矩阵
                M = cv2.getAffineTransform(src_tri_adj, dst_tri_adj)

                # 创建三角形mask
                mask = np.zeros((src_y_max - src_y_min, src_x_max - src_x_min), dtype=np.uint8)
                pts = src_tri_adj.astype(np.int32).reshape(-1, 1, 2)
                cv2.fillConvexPoly(mask, pts, 255)

                # 对图像和mask做仿射变换
                dst_w = dst_x_max - dst_x_min
                dst_h = dst_y_max - dst_y_min

                warped_img = cv2.warpAffine(src_region, M, (dst_w, dst_h),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

                warped_mask = cv2.warpAffine(mask, M, (dst_w, dst_h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)

                # 转换为BGRA并设置alpha
                warped_bgra = cv2.cvtColor(warped_img, cv2.COLOR_BGR2BGRA)
                warped_bgra[:, :, 3] = warped_mask

                # Alpha混合到结果
                roi = result[dst_y_min:dst_y_max, dst_x_min:dst_x_max]
                alpha = warped_mask.astype(np.float32) / 255.0

                for c in range(4):
                    roi[:, :, c] = (warped_bgra[:, :, c] * alpha +
                                    roi[:, :, c] * (1 - alpha)).astype(np.uint8)

    print(f"Warp完成: 共{(gridX-1)*(gridY-1)}个单元, {warped_triangles}个三角形被变换")

    # 转换回BGR
    result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    return result_bgr


@app.route('/api/warp', methods=['POST'])
def warp_image():
    """
    执行图像Warp处理
    请求体:
    {
        "image": "base64编码的图像数据",
        "gcps": [
            {"sourceX": 100, "sourceY": 100, "targetX": 120, "targetY": 110},
            ...
        ],
        "gridX": 横向控制点数量,
        "gridY": 纵向控制点数量,
        "width": 图像宽度,
        "height": 图像高度
    }
    """
    try:
        data = request.json

        # 解码图像
        image_data = data['image']
        image_bytes = base64.b64decode(image_data.split(',')[1])

        # 使用PIL打开图像
        pil_image = Image.open(io.BytesIO(image_bytes))

        # 转换为RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        width = pil_image.width
        height = pil_image.height

        # 转换为numpy数组（RGB顺序，OpenCV需要BGR）
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 获取控制点和网格参数
        gcps = data['gcps']
        gridX = data.get('gridX', 3)
        gridY = data.get('gridY', 3)

        # 准备源点和目标点
        source_points = np.array([[g['sourceX'], g['sourceY']] for g in gcps], dtype=np.float32)
        target_points = np.array([[g['targetX'], g['targetY']] for g in gcps], dtype=np.float32)

        # 调试：打印控制点信息
        print(f"\n图像尺寸: {width}x{height}")
        print(f"网格: {gridX} x {gridY}")
        print(f"控制点数量: {len(source_points)}")

        # 打印前几个控制点的变化
        print("\n控制点变化情况:")
        for i in range(min(5, len(source_points))):
            src = source_points[i]
            dst = target_points[i]
            diff = np.linalg.norm(src - dst)
            print(f"  点{i}: 源({src[0]:.1f},{src[1]:.1f}) -> 目标({dst[0]:.1f},{dst[1]:.1f}), 移动={diff:.1f}px")

        # 检查控制点是否有变化
        if np.allclose(source_points, target_points, atol=0.5):
            # 没有变化，直接返回原图
            print("控制点无变化（阈值0.5px），返回原图")
            result_image = img_bgr
        else:
            # 执行基于网格的warp变换
            result_image = grid_warp_image(img_bgr, source_points, target_points, gridX, gridY, (height, width))

        # 转换回RGB
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        result_pil = Image.fromarray(result_rgb)

        # 转换为base64
        buffered = io.BytesIO()
        result_pil.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_str,
            'width': result_pil.width,
            'height': result_pil.height
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("图像几何校正服务器")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
