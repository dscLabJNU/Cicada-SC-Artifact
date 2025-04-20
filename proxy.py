from flask import Flask, request, jsonify, Response
from gevent.pywsgi import WSGIServer
import torch
import io
import os

from strategy import ModelCacheStrategy, TetrisStrategy
from model_loader import load_model_from_source
from logger_config import get_logger

# 获取 logger
logger = get_logger(__name__)

app = Flask(__name__)

# 当前使用的策略
current_strategy: ModelCacheStrategy = None


def init_strategy(strategy_name: str, use_persistent_cache: bool = False) -> None:
    """初始化指定的缓存策略"""
    global current_strategy

    # 直接根据策略名称初始化对应的策略类
    if strategy_name == 'tetris':
        if use_persistent_cache:
            current_strategy = TetrisStrategy(persistent='persistent_cache')
        else:
            current_strategy = TetrisStrategy(
                cache_size=100 * 1024 * 1024 * 1024)  # 100GB -> simulate infinite cache
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    current_strategy.initialize()
    logger.info(
        f"Initialized {strategy_name} strategy with {'persistent' if use_persistent_cache else 'in-memory'} cache.")


@app.route('/model_structure', methods=['GET'])
def model_structure():
    """获取模型结构接口"""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({"error": "model_name parameter is required."}), 400
    dedup = data.get('dedup', False)
    try:
        if dedup:
            model = current_strategy.get_model_structure(model_name)
        else:
            model = load_model_from_source(model_name)
    except Exception as e:
        logger.error(f"Error getting model structure: {e}")
        return jsonify({"error": str(e)}), 500

    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype='application/octet-stream',
        headers={"Message": f"Model structure retrieved successfully."}
    )


@app.route('/model_load', methods=['GET'])
def model_load():
    """加载模型接口"""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400
    model_name = data.get('model_name')
    weights_key = data.get('weights_key', "")

    if not model_name:
        return jsonify({"error": "model_name parameter is required."}), 400

    try:
        # 首先检查模型是否已在缓存中
        if model_name in current_strategy.models_manifest:
            logger.info(
                f"Model '{model_name}' found in cache, returning cached state_dict")
            state_dict = current_strategy.get_model_state_dict(model_name)
        else:
            # 模型不在缓存中，需要加载
            logger.info(
                f"Loading model '{model_name}' with weights_key '{weights_key}'")
            model = load_model_from_source(model_name, weights_key)
            current_strategy.add_model(model, model_name, weights_key)
            state_dict = current_strategy.get_model_state_dict(model_name)

        # 序列化
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        return Response(
            buffer.getvalue(),
            mimetype='application/octet-stream',
            headers={"Message": f"Model '{model_name}' {'loaded from cache' if model_name in current_strategy.models_manifest else 'loaded and cached'} successfully."}
        )

    except Exception as e:
        import traceback
        logger.error(f"Error loading model: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/layer_state', methods=['GET'])
def layer_state():
    """获取层状态接口"""
    model_name = request.args.get('model_name')
    layer_name = request.args.get('layer_name')

    if not model_name or not layer_name:
        return jsonify({"error": "model_name and layer_name parameters are required."}), 400

    try:
        param_tensor = current_strategy.get_layer_state(model_name, layer_name)

        buffer = io.BytesIO()
        torch.save(param_tensor, buffer)
        buffer.seek(0)

        return Response(
            buffer.getvalue(),
            mimetype='application/octet-stream',
            headers={
                "Message": f"Layer '{layer_name}' data retrieved successfully."}
        )

    except Exception as e:
        logger.error(f"Error getting layer state: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    strategy_name = os.getenv('CACHE_STRATEGY', 'tetris')
    use_persistent_cache = os.getenv(
        'USE_PERSISTENT_CACHE', 'false').lower() == 'true'
    init_strategy(strategy_name, use_persistent_cache)

    # 启动服务器
    server = WSGIServer(("0.0.0.0", 8888), app)
    logger.info(
        f"Starting Flask server on port 8888 with {strategy_name} strategy...")
    server.serve_forever()
