from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import uuid
import os
from typing import Optional
import uvicorn
from config.config_loader import load_config
from config.logger import setup_logging
from core.utils.modules_initialize import initialize_modules
from core.connection import ConnectionHandler

app = FastAPI(title="Voice QA System API", version="1.0.0")
# 添加 CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["*"],  # 暴露所有响应头
)
# 全局配置 pip install fastapi uvicorn
config = None
modules = {}
logger = None

# 挂载静态文件目录，供返回的 audio_url 访问
app.mount("/audio", StaticFiles(directory="tmp"), name="audio")


async def _schedule_delete(path: str, delay: int = 30):
    """延迟删除生成的音频文件（后台任务）"""
    try:
        await asyncio.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
            if logger:
                logger.info(f"已删除临时音频文件: {path}")
    except Exception as e:
        if logger:
            logger.error(f"删除临时音频文件失败: {path} -> {e}")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global config, modules, logger
    try:
        config = load_config()
        logger = setup_logging()
        modules = initialize_modules(
            logger,
            config,
            False,
            "ASR" in config["selected_module"],
            "LLM" in config["selected_module"],
            "TTS" in config["selected_module"],
            False,
            False,
        )
        os.makedirs("tmp", exist_ok=True)
        logger.info("静态文件目录已挂载: /audio -> tmp/")
        logger.info("FastAPI: 系统初始化完成")
    except Exception as e:
        if logger:
            logger.error(f"FastAPI: 系统初始化失败: {e}")
        raise

async def process_voice_query(audio_file_path: str, session_id: str = None):
    """异步处理语音查询"""
    try:
        asr = modules.get("asr")
        llm = modules.get("llm")
        tts = modules.get("tts")
        
        if not all([asr, llm, tts]):
            raise Exception("系统模块未正确初始化")
        
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"开始处理音频文件: {audio_file_path}")

        # 1. 语音识别
        text, _ = await asr.speech_to_text_from_audio_file(audio_file_path, session_id)
        logger.info(f"语音识别结果: {text}")

        if not text:
            raise Exception("语音识别失败")
        
        # 2. 对话处理
        conn = ConnectionHandler(config, tts, asr, llm)
        logger.info(f"开始对话，用户输入: {text}")
        result = conn.chat(text)
        logger.info(f"对话结果: {result}")

        if not result:
            raise Exception("对话处理失败")

        # 3. TTS 合成并返回可访问的 audio_url
        audio_file = None
        try:
            audio_file = conn.tts.generate_filename()
            logger.info(f"生成 TTS 音频文件: {audio_file}")
            
            # 调用 TTS 合成（异步）
            if asyncio.iscoroutinefunction(conn.tts.text_to_speak):
                await conn.tts.text_to_speak(conn.tts_MessageText, audio_file)
            else:
                # 如果实现是同步的，放到线程池中执行
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, conn.tts.text_to_speak, conn.tts_MessageText, audio_file)

            audio_url = f"/audio/{os.path.basename(audio_file)}"

            # 根据配置决定是否延迟删除
            try:
                delete_audio = config.get("delete_audio", True)
            except Exception:
                delete_audio = True
            if delete_audio:
                asyncio.create_task(_schedule_delete(audio_file, delay=300))

        except Exception as e:
            if logger:
                logger.error(f"TTS 合成失败: {e}")
            audio_url = None

        return {
            "session_id": session_id,
            "recognized_text": text,
            "response_text": conn.tts_MessageText,
            "audio_url": audio_url,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"FastAPI: 处理语音查询失败: {e}")
        raise

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "service": "Voice QA System - FastAPI",
        "timestamp": asyncio.get_event_loop().time()
    }

@app.post("/api/v1/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """语音对话接口"""
    try:
        # 保存临时文件
        temp_filename = f"tmp/{uuid.uuid4()}_{audio.filename}"
        os.makedirs("tmp", exist_ok=True)
        
        with open(temp_filename, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # 处理请求
        if not session_id:
            session_id = str(uuid.uuid4())
            
        result = await process_voice_query(temp_filename, session_id)
        
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return result
        
    except Exception as e:
        # 清理临时文件
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        logger.error(f"FastAPI: 语音对话处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
class TextRequest(BaseModel):
    text: str

@app.post("/api/v1/text-chat")
async def text_chat(request: TextRequest):
    """文本对话接口 - 从请求体获取文本"""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="文本内容为空")
        
        # 获取模块
        llm = modules.get("llm")
        tts = modules.get("tts")
        
        if not all([llm, tts]):
            raise HTTPException(status_code=500, detail="系统模块未正确初始化")
        
        # 处理对话
        conn = ConnectionHandler(config, tts, None, llm)
        result = conn.chat(text)
        
        if not result:
            raise HTTPException(status_code=500, detail="对话处理失败")
         # 3. TTS 合成并返回可访问的 audio_url
        audio_file = None
        try:
            audio_file = conn.tts.generate_filename()
            logger.info(f"生成 TTS 音频文件: {audio_file}")
            
            # 调用 TTS 合成（异步）
            if asyncio.iscoroutinefunction(conn.tts.text_to_speak):
                await conn.tts.text_to_speak(conn.tts_MessageText, audio_file)
            else:
                # 如果实现是同步的，放到线程池中执行
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, conn.tts.text_to_speak, conn.tts_MessageText, audio_file)

            audio_url = f"/audio/{os.path.basename(audio_file)}"

            # 根据配置决定是否延迟删除
            try:
                delete_audio = config.get("delete_audio", True)
            except Exception:
                delete_audio = True
            if delete_audio:
                asyncio.create_task(_schedule_delete(audio_file, delay=300))

        except Exception as e:
            if logger:
                logger.error(f"TTS 合成失败: {e}")
            audio_url = None

        return {
            "response_text": conn.tts_MessageText,
            "audio_url": audio_url,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FastAPI: 文本对话处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)