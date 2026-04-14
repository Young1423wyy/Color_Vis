from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
import requests
import os
from collections import Counter

app = Flask(__name__)

# 将RGB颜色转换为Hex字符串供前端使用
def rgb2hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(max(0, min(rgb[0], 255))), 
                                        int(max(0, min(rgb[1], 255))), 
                                        int(max(0, min(rgb[2], 255))))

@app.route('/')
def index():
    # 获取可用的图片列表
    images = os.listdir('static/images')
    return render_template('index.html', images=images)

@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.json
    img_name = data.get('image')
    k = int(data.get('k', 5))
    color_space = data.get('color_space', 'RGB')
    
    img_path = os.path.join('static', 'images', img_name)
    
    # 读取图片并缩放以加快聚类速度
    img = Image.open(img_path).convert('RGB')
    img = img.resize((150, 150))
    pixels = np.array(img)
    
    # 改变数据形状以适应聚类模型
    pixels_reshape = pixels.reshape(-1, 3)
    
    if color_space == 'LAB':
        # 将RGB转换到[0,1]然后转为LAB
        pixels_lab = color.rgb2lab(pixels_reshape.reshape(150, 150, 3) / 255.0).reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels_lab)
        # 将聚类中心转回RGB用于显示
        centers_lab = kmeans.cluster_centers_.reshape(k, 1, 3)
        centers_rgb = color.lab2rgb(centers_lab).reshape(k, 3) * 255.0
    else:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels_reshape)
        centers_rgb = kmeans.cluster_centers_

    # 统计每一类的像素数量
    counts = Counter(kmeans.labels_)
    
    results = []
    for i in range(k):
        results.append({
            'color': rgb2hex(centers_rgb[i]),
            'count': counts[i]
        })
        
    return jsonify({'status': 'success', 'data': results})

@app.route('/check_harmony', methods=['POST'])
def check_harmony():
    colors = request.json.get('colors', [])
    color_str = ", ".join(colors)
    
    # API和Key
    API_URL = "https://api.openai-proxy.org/v1/chat/completions" 
    API_KEY = "sk-84rhTYEo1etKo7GF4Y2YaCS5B7iCDQLXy0ysjgH6IyTCAvri" 
    
    prompt = f"我有以下这些颜色：{color_str}。请以专业设计师的角度，简短判断一下这些颜色搭配在一起是否和谐？请用中文回答并在100字以内。"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo", 
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
            return jsonify({'status': 'success', 'message': reply})
        else:
            return jsonify({'status': 'error', 'message': 'API调用失败'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)